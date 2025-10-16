"""
Système de génération d'état de l'art scientifique
Utilise Semantic Scholar API et un LLM (Gemini par défaut)
"""

import requests
import time
from typing import List, Dict, Optional
import streamlit as st
from dataclasses import dataclass
import google.generativeai as genai
import warnings

# Supprime les warnings non critiques
warnings.filterwarnings('ignore')

# ============================================================================
# SYSTÈME D'AUTHENTIFICATION
# ============================================================================

def check_auth():
    """
    Système d'authentification simple.
    Retourne True si l'utilisateur est authentifié.
    """
    def password_entered():
        """Vérifie si le mot de passe est correct"""
        # Récupère le mot de passe depuis les secrets Streamlit
        if "APP_PASSWORD" in st.secrets:
            correct_password = st.secrets["APP_PASSWORD"]
        else:
            # Mot de passe par défaut pour test local
            correct_password = "demo123"
        
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Supprime le mot de passe de la session
        else:
            st.session_state["password_correct"] = False

    # Première visite
    if "password_correct" not in st.session_state:
        st.markdown("### 🔐 Accès restreint")
        st.text_input(
            "Mot de passe", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.info("💡 Contactez l'administrateur pour obtenir le mot de passe")
        return False
    
    # Mot de passe incorrect
    elif not st.session_state["password_correct"]:
        st.markdown("### 🔐 Accès restreint")
        st.text_input(
            "Mot de passe", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😕 Mot de passe incorrect")
        return False
    
    # Authentifié avec succès
    else:
        return True

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration centralisée pour faciliter le changement de LLM"""
    semantic_scholar_api_url: str = "https://api.semanticscholar.org/graph/v1"
    max_articles: int = 15  # Nombre maximum d'articles à récupérer
    llm_provider: str = "gemini"  # Facilite le changement futur
    gemini_model: str = "gemini-2.5-flash"  # Ou essayez "models/gemini-2.5-flash"
    

# ============================================================================
# CLASSE LLM ABSTRACTION (facilite le changement de LLM)
# ============================================================================

class LLMProvider:
    """
    Classe abstraite pour interagir avec différents LLM.
    Permet de changer facilement de provider (Gemini, GPT, Claude, etc.)
    """
    
    def __init__(self, api_key: str, provider: str = "gemini", model_name: str = "gemini-2.5-flash"):
        self.provider = provider
        self.api_key = api_key
        
        if provider == "gemini":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
    
    def generate(self, prompt: str) -> str:
        """
        Génère une réponse à partir d'un prompt.
        Cette méthode peut être adaptée pour d'autres LLM.
        """
        try:
            if self.provider == "gemini":
                response = self.model.generate_content(prompt)
                # Vérification de la réponse
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'parts'):
                    # Parfois la réponse est dans parts
                    return ''.join([part.text for part in response.parts])
                else:
                    st.error("⚠️ Le LLM a retourné une réponse vide ou bloquée")
                    return ""
            # Ajouter d'autres providers ici (GPT, Claude, etc.)
            else:
                raise ValueError(f"Provider {self.provider} non supporté")
        except Exception as e:
            st.error(f"Erreur LLM: {str(e)}")
            # Affiche plus d'infos pour déboguer
            if hasattr(e, 'message'):
                st.error(f"Détails: {e.message}")
            return ""


# ============================================================================
# SEMANTIC SCHOLAR API
# ============================================================================

class SemanticScholarAPI:
    """Gère les interactions avec l'API Semantic Scholar"""
    
    def __init__(self, config: Config, api_key: Optional[str] = None):
        self.config = config
        self.base_url = config.semantic_scholar_api_url
        self.api_key = api_key
        self.headers = {}
        
        # Si une clé API est fournie, l'ajouter aux headers
        if self.api_key:
            self.headers['x-api-key'] = self.api_key
    
    def extract_keywords(self, question: str, llm: LLMProvider) -> List[str]:
        """
        Utilise le LLM pour extraire les mots-clés pertinents de la question.
        Améliore la précision de la recherche sur Semantic Scholar.
        """
        prompt = f"""
        Analyse la question technique suivante et extrais 3-5 mots-clés ou expressions clés 
        en anglais pour rechercher des articles scientifiques pertinents.
        
        Question: {question}
        
        Réponds uniquement avec les mots-clés séparés par des virgules, sans numérotation.
        Exemple: deep learning, neural networks, image classification
        """
        
        try:
            keywords_text = llm.generate(prompt)
            if not keywords_text or keywords_text.strip() == "":
                # Fallback : utilise la question directement
                st.warning("⚠️ Impossible d'extraire les mots-clés, utilisation de la question complète")
                return [question]
            # Nettoie et sépare les mots-clés
            keywords = [k.strip() for k in keywords_text.split(',')]
            return keywords[:5]  # Limite à 5 mots-clés
        except Exception as e:
            st.error(f"Erreur lors de l'extraction des mots-clés: {str(e)}")
            # Fallback : utilise la question directement
            return [question]
    
    def search_papers(self, query: str, limit: int = 15) -> List[Dict]:
        """
        Recherche des articles sur Semantic Scholar.
        Retourne une liste d'articles avec leurs métadonnées.
        """
        url = f"{self.base_url}/paper/search"
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,abstract,authors,year,citationCount,publicationDate,url,paperId,tldr,openAccessPdf'
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            # Affiche des infos sur l'utilisation de la clé API
            if self.api_key:
                st.success("✅ Clé API Semantic Scholar utilisée")
            else:
                st.info("ℹ️ Recherche sans clé API (limites réduites)")
            
            return data.get('data', [])
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                st.error("⚠️ Limite de taux atteinte. Veuillez patienter quelques secondes et réessayer.")
                st.info("💡 Astuce : Réduisez le nombre d'articles ou attendez 1 minute avant de relancer.")
            else:
                st.error(f"Erreur API Semantic Scholar: {str(e)}")
            return []
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur API Semantic Scholar: {str(e)}")
            return []
    
    def get_paper_details(self, paper_id: str) -> Optional[Dict]:
        """
        Récupère les détails complets d'un article spécifique.
        Utilisé pour obtenir plus d'informations si nécessaire.
        """
        url = f"{self.base_url}/paper/{paper_id}"
        params = {
            'fields': 'title,abstract,authors,year,citationCount,references,citations,url,tldr,openAccessPdf,externalIds'
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la récupération de l'article: {str(e)}")
            return None
    
    def fetch_paper_content(self, paper: Dict) -> Optional[str]:
        """
        Tente de récupérer le contenu complet d'un article pour extraire un résumé.
        Essaie plusieurs sources : TLDR de Semantic Scholar, PDF Open Access, etc.
        """
        paper_id = paper.get('paperId')
        if not paper_id:
            return None
        
        # Récupère les détails complets incluant TLDR et PDF
        details = self.get_paper_details(paper_id)
        if not details:
            return None
        
        # 1. Essaye d'abord le TLDR (résumé automatique de Semantic Scholar)
        if details.get('tldr') and details['tldr'].get('text'):
            return details['tldr']['text']
        
        # 2. Vérifie si un PDF Open Access est disponible
        open_access = details.get('openAccessPdf')
        if open_access and open_access.get('url'):
            st.info(f"📄 PDF Open Access disponible pour: {paper.get('title', '')[:50]}...")
            return f"PDF disponible à: {open_access['url']}"
        
        return None


# ============================================================================
# GÉNÉRATEUR D'ÉTAT DE L'ART
# ============================================================================

class LiteratureReviewGenerator:
    """Génère l'état de l'art à partir des articles sélectionnés"""
    
    def __init__(self, llm: LLMProvider):
        self.llm = llm
    
    def summarize_paper(self, paper: Dict, semantic_api: 'SemanticScholarAPI') -> str:
        """
        Génère un résumé court d'un article pour la liste de validation.
        Tente d'abord d'utiliser l'abstract, puis le TLDR (déjà récupéré lors de la recherche).
        """
        title = paper.get('title', 'Sans titre')
        abstract = paper.get('abstract', None)
        year = paper.get('year', 'N/A')
        
        # 1. Si abstract disponible, l'utiliser
        if abstract and abstract.strip():
            prompt = f"""
            Résume cet article scientifique en 2-3 phrases maximum, en français.
            Concentre-toi sur la contribution principale et les résultats clés.
            
            Titre: {title}
            Année: {year}
            Résumé: {abstract[:1000]}
            """
            
            summary = self.llm.generate(prompt)
            if summary and summary.strip():
                return summary
            return abstract[:300] + "..."
        
        # 2. Si pas d'abstract, essayer le TLDR de Semantic Scholar (déjà dans les données)
        tldr = paper.get('tldr')
        if tldr and tldr.get('text'):
            tldr_text = tldr['text']
            prompt = f"""
            Traduis et reformule ce résumé en français (2-3 phrases):
            
            Titre: {title}
            TLDR: {tldr_text}
            """
            summary = self.llm.generate(prompt)
            if summary and summary.strip():
                return summary
            return tldr_text
        
        # 3. Vérifier si un PDF Open Access est disponible (déjà dans les données)
        open_access = paper.get('openAccessPdf')
        if open_access and open_access.get('url'):
            return f"⚠️ Résumé non disponible dans l'API. PDF Open Access disponible à: {open_access['url']}"
        
        # 4. Si rien n'est disponible, indiquer clairement
        paper_url = paper.get('url', '')
        if paper_url:
            return f"⚠️ Résumé non disponible. Consultez l'article complet: {paper_url}"
        return f"⚠️ Résumé non disponible pour cet article."
    
    def generate_full_review(self, papers: List[Dict], question: str) -> str:
        """
        Génère l'état de l'art complet (environ 2 pages).
        Structure: Introduction, Travaux existants, Limitations, Perspectives.
        """
        # Prépare les informations sur les articles
        papers_info = []
        for i, paper in enumerate(papers, 1):
            authors_list = paper.get('authors', [])
            authors = ', '.join([a.get('name', '') for a in authors_list[:3]])
            if len(authors_list) > 3:
                authors += " et al."
            
            # Utilise l'abstract, ou le TLDR, ou le summary généré
            abstract = paper.get('abstract', None)
            if abstract and abstract.strip():
                abstract_text = abstract[:800]
            elif paper.get('tldr') and paper.get('tldr').get('text'):
                abstract_text = paper['tldr']['text']
            else:
                # Utilise le summary qui a été généré par summarize_paper
                abstract_text = paper.get('summary', 'Non disponible')[:800]
            
            info = f"""
            Article {i}:
            - Titre: {paper.get('title', 'N/A')}
            - Auteurs: {authors}
            - Année: {paper.get('year', 'N/A')}
            - Citations: {paper.get('citationCount', 0)}
            - Résumé: {abstract_text}
            """
            papers_info.append(info)
        
        papers_text = "\n\n".join(papers_info)
        
        prompt = f"""
        Tu es un chercheur expert. Rédige un état de l'art scientifique complet et structuré 
        en français sur la question suivante: "{question}"
        
        Utilise les articles scientifiques ci-dessous comme base.
        
        {papers_text}
        
        Structure ton état de l'art en QUATRE sections obligatoires:
        
        1. INTRODUCTION (1 paragraphe)
           - Contextualise la problématique
           - Explique l'importance du sujet
        
        2. TRAVAUX EXISTANTS (60% du contenu)
           - Présente les contributions majeures de chaque article
           - Organise par thèmes ou approches
           - Cite les auteurs et années entre parenthèses
           - Compare les méthodes et résultats
        
        3. LIMITATIONS (20% du contenu)
           - Identifie les limites des approches actuelles
           - Points non résolus dans la littérature
           - Contradictions ou débats
        
        4. PERSPECTIVES ET AXES DE RECHERCHE (20% du contenu)
           - Directions futures prometteuses
           - Questions ouvertes
           - Recommandations pour de futurs travaux
        
        CONSIGNES:
        - Longueur: environ 2 pages (1000-1200 mots)
        - Style: académique mais accessible
        - Synthèse: regroupe les travaux similaires
        - Citations: mentionne auteurs et années
        - Objectivité: présente les forces ET faiblesses
        
        Rédige maintenant l'état de l'art complet.
        """
        
        review = self.llm.generate(prompt)
        
        # Vérification de la génération
        if not review or review.strip() == "":
            st.error("⚠️ La génération de l'état de l'art a échoué")
            return "Erreur lors de la génération. Veuillez réessayer ou changer de modèle LLM."
        
        return review


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def main():
    st.set_page_config(
        page_title="Générateur d'État de l'Art",
        page_icon="📚",
        layout="wide"
    )

    # ========================================================================
    # AUTHENTIFICATION - PREMIÈRE CHOSE À VÉRIFIER
    # ========================================================================
    if not check_auth():
        st.stop()  # Arrête tout si pas authentifié
    
    st.title("📚 Générateur d'État de l'Art Scientifique")
    st.markdown("*Propulsé par Semantic Scholar et Gemini*")
    
    # ========================================================================
    # SIDEBAR: Configuration
    # ========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Clé API Gemini
        api_key = st.text_input(
            "Clé API Gemini",
            type="password",
            help="Obtenir une clé sur https://makersuite.google.com/app/apikey"
        )
        
        # Clé API Semantic Scholar
        ss_api_key = st.text_input(
            "Clé API Semantic Scholar (optionnelle)",
            type="password",
            help="Obtenir une clé sur https://www.semanticscholar.org/product/api"
        )
        
        # Sélection du modèle
        model_choice = st.selectbox(
            "Modèle Gemini",
            options=[ 
                "models/gemini-2.5-flash",
                "models/gemini-2.5-pro"
            ],
            index=0,
            help="Si un modèle ne fonctionne pas, essayez-en un autre"
        )
        
        # Nombre d'articles
        max_articles = st.slider(
            "Nombre d'articles à rechercher",
            min_value=5,
            max_value=30,
            value=15
        )
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. Entrez votre clé API Gemini
        2. Posez votre question technique
        3. Validez les articles trouvés
        4. Générez l'état de l'art
        """)
    
    # Vérifie la présence de la clé API
    if not api_key:
        st.warning("⚠️ Veuillez entrer votre clé API Gemini dans la barre latérale")
        return
    
    # ========================================================================
    # INITIALISATION
    # ========================================================================
    config = Config(max_articles=max_articles)
    llm = LLMProvider(api_key=api_key, model_name=model_choice)
    semantic_api = SemanticScholarAPI(config, api_key=ss_api_key if ss_api_key else None)
    review_generator = LiteratureReviewGenerator(llm)
    
    # Initialise l'état de session
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    if 'question' not in st.session_state:
        st.session_state.question = ""
    if 'search_done' not in st.session_state:
        st.session_state.search_done = False
    
    # ========================================================================
    # ÉTAPE 1: Question de l'utilisateur
    # ========================================================================
    st.header("1️⃣ Posez votre question")
    
    question = st.text_area(
        "Question technique",
        placeholder="Ex: Quelles sont les méthodes d'apprentissage par renforcement pour la robotique ?",
        height=100
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Rechercher", type="primary")
    with col2:
        if st.session_state.search_done:
            if st.button("🔄 Nouvelle recherche"):
                st.session_state.papers = []
                st.session_state.search_done = False
                st.rerun()
    
    # ========================================================================
    # ÉTAPE 2: Recherche et affichage des articles
    # ========================================================================
    if search_button and question:
        st.session_state.question = question
        
        with st.spinner("🔎 Extraction des mots-clés..."):
            keywords = semantic_api.extract_keywords(question, llm)
            st.info(f"**Mots-clés identifiés:** {', '.join(keywords)}")
        
        with st.spinner("📖 Recherche d'articles sur Semantic Scholar..."):
            # Recherche avec les mots-clés combinés
            query = ' '.join(keywords)
            papers = semantic_api.search_papers(query, limit=max_articles)
            
            if not papers:
                st.error("Aucun article trouvé. Essayez de reformuler votre question.")
                return
            
            st.success(f"✅ {len(papers)} articles trouvés !")
        
        # Génère les résumés
        with st.spinner("✍️ Génération des résumés..."):
            for i, paper in enumerate(papers):
                # Ajoute un délai pour respecter les limites de taux de l'API
                if i > 0:
                    time.sleep(1.0)  # Pause de 1 seconde entre chaque résumé
                
                try:
                    paper['summary'] = review_generator.summarize_paper(paper, semantic_api)
                except Exception as e:
                    st.warning(f"⚠️ Erreur lors de la génération du résumé pour '{paper.get('title', 'Article')[:50]}...': {str(e)}")
                    paper['summary'] = f"Erreur de génération. Consultez l'article: {paper.get('url', 'URL non disponible')}"
        
        st.session_state.papers = papers
        st.session_state.search_done = True
    
    # ========================================================================
    # AFFICHAGE DES RÉSULTATS
    # ========================================================================
    if st.session_state.search_done and st.session_state.papers:
        st.header("2️⃣ Articles trouvés")
        st.markdown(f"**Question:** {st.session_state.question}")
        
        # Sélection des articles
        st.subheader("Sélectionnez les articles à inclure dans l'état de l'art")
        
        selected_papers = []
        for i, paper in enumerate(st.session_state.papers):
            with st.expander(
                f"📄 {paper.get('title', 'Sans titre')} ({paper.get('year', 'N/A')})",
                expanded=(i < 3)  # Les 3 premiers sont ouverts par défaut
            ):
                # Checkbox de sélection
                is_selected = st.checkbox(
                    "Inclure dans l'état de l'art",
                    value=True,  # Tous sélectionnés par défaut
                    key=f"select_{i}"
                )
                
                # Informations sur l'article
                authors = paper.get('authors', [])
                author_names = ', '.join([a.get('name', '') for a in authors[:3]])
                if len(authors) > 3:
                    author_names += f" et al."
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Auteurs:** {author_names}")
                    st.markdown(f"**Année:** {paper.get('year', 'N/A')}")
                with col2:
                    st.markdown(f"**Citations:** {paper.get('citationCount', 0)}")
                    if paper.get('url'):
                        st.markdown(f"[🔗 Lien]({paper['url']})")
                
                st.markdown("**Résumé:**")
                st.write(paper.get('summary', 'Résumé non disponible'))
                
                if is_selected:
                    selected_papers.append(paper)
        
        st.markdown(f"**{len(selected_papers)} articles sélectionnés**")
        
        # ====================================================================
        # ÉTAPE 3: Génération de l'état de l'art
        # ====================================================================
        st.header("3️⃣ Génération de l'état de l'art")
        
        if st.button("📝 Générer l'état de l'art complet", type="primary", disabled=len(selected_papers) == 0):
            if len(selected_papers) == 0:
                st.warning("⚠️ Veuillez sélectionner au moins un article")
            else:
                with st.spinner("✨ Génération de l'état de l'art en cours... (cela peut prendre 1-2 minutes)"):
                    try:
                        review = review_generator.generate_full_review(
                            selected_papers,
                            st.session_state.question
                        )
                        
                        if review and review.strip() != "":
                            st.success("✅ État de l'art généré avec succès !")
                            
                            # Affichage de l'état de l'art
                            st.markdown("---")
                            st.markdown("## 📑 État de l'art")
                            st.markdown(review)
                            
                            # Bouton de téléchargement
                            st.download_button(
                                label="💾 Télécharger l'état de l'art",
                                data=review,
                                file_name="etat_de_lart.txt",
                                mime="text/plain"
                            )
                        else:
                            st.error("❌ La génération a échoué. Essayez avec un autre modèle LLM.")
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération : {str(e)}")
                        st.info("💡 Essayez de sélectionner moins d'articles ou changez de modèle LLM")


if __name__ == "__main__":

    main()
