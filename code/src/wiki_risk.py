import requests
from datetime import datetime
from typing import Dict, Optional, List

class EntityRiskScorer:
    def __init__(self, news_api_key: str):
        """Initialize with your NewsAPI key"""
        self.news_api_key = news_api_key
        self.wiki_user_agent = "EntityRiskScorer/1.0 (contact@example.com)"
        
        # API endpoints
        self.wiki_api = "https://en.wikipedia.org/w/api.php"
        self.wikidata_api = "https://www.wikidata.org/w/api.php"
        self.news_api = "https://newsapi.org/v2/everything"
        
        # Risk configuration
        self.high_risk_jurisdictions = ['panama', 'cayman', 'bvi', 'virgin islands', 
                                      'seychelles', 'malta', 'cyprus', 'mauritius']
        self.shell_company_types = ['Q201818', 'Q7278']  # Shell company and corporation types
        self.high_risk_industries = ['Q188569', 'Q131645']  # Offshore finance, gambling
        
        # Configuration
        self.max_news_articles = 10  # Conservative limit for free tier
        self.request_timeout = 10  # seconds

    def get_risk_score(self, entity_name: str, jurisdiction: str = None) -> Dict:
        """Calculate comprehensive risk score (0-100) for an entity"""
        print(f"\nAssessing risk for: {entity_name} ({jurisdiction or 'no jurisdiction'})")
        
        # Data collection from all sources
        wiki_data = self._get_wikipedia_data(entity_name)
        wikidata_info = self._query_wikidata(entity_name)
        news_data = self._get_news_data(entity_name, jurisdiction)
        
        # Risk assessment components
        risk_components = {
            'entity_structure': self._calc_entity_risk(wikidata_info),
            'jurisdiction': self._calc_location_risk(jurisdiction, wikidata_info),
            'reputation': self._calc_reputation_risk(wiki_data, news_data),
            'financial_transparency': self._calc_financial_risk(wikidata_info)
        }
        
        # Calculate weighted score
        total_score = sum(
            comp['score'] * comp['weight'] 
            for comp in risk_components.values()
        )
        
        # Confidence score based on data quality
        confidence = self._calc_confidence(wiki_data, wikidata_info, news_data)
        
        return {
            'entity': entity_name,
            'jurisdiction': jurisdiction,
            'risk_score': min(100, int(total_score)),
            'risk_level': self._get_risk_level(total_score),
            'confidence': confidence,
            'risk_breakdown': risk_components,
            'evidence': {
                'wikipedia': wiki_data.get('url'),
                'wikidata': wikidata_info,
                'top_news': [a['title'] for a in news_data.get('articles', [])[:3]]
            },
            'timestamp': datetime.now().isoformat()
        }

    def _get_wikipedia_data(self, entity_name: str) -> Dict:
        """Fetch Wikipedia page data"""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': entity_name,
            'prop': 'extracts|pageprops|info',
            'inprop': 'url',
            'exintro': True,
            'explaintext': True
        }
        
        try:
            headers = {'User-Agent': self.wiki_user_agent}
            response = requests.get(
                self.wiki_api, 
                params=params, 
                headers=headers,
                timeout=self.request_timeout
            )
            response.raise_for_status()
            page = next(iter(response.json()['query']['pages'].values()))
            
            return {
                'exists': 'missing' not in page,
                'title': page.get('title'),
                'url': f"https://en.wikipedia.org/?curid={page.get('pageid', '')}",
                'extract': page.get('extract', ''),
                'controversial': self._detect_controversy(page)
            }
        except Exception as e:
            print(f"Wikipedia API error: {e}")
            return {'exists': False}

    def _query_wikidata(self, entity_name: str) -> Optional[Dict]:
        """Query Wikidata for entity information"""
        try:
            # Step 1: Search for entity
            search_params = {
                'action': 'wbsearchentities',
                'search': entity_name,
                'language': 'en',
                'format': 'json'
            }
            search_response = requests.get(
                self.wikidata_api,
                params=search_params,
                headers={'User-Agent': self.wiki_user_agent},
                timeout=self.request_timeout
            )
            search_response.raise_for_status()
            search_data = search_response.json()
            
            if not search_data.get('search'):
                return None
                
            # Step 2: Get entity details
            qid = search_data['search'][0]['id']
            entity_params = {
                'action': 'wbgetentities',
                'ids': qid,
                'props': 'claims|descriptions',
                'format': 'json'
            }
            entity_response = requests.get(
                self.wikidata_api,
                params=entity_params,
                headers={'User-Agent': self.wiki_user_agent},
                timeout=self.request_timeout
            )
            entity_response.raise_for_status()
            entity_data = entity_response.json()
            claims = entity_data.get('entities', {}).get(qid, {}).get('claims', {})
            
            return {
                'id': qid,
                'name': search_data['search'][0].get('label'),
                'description': search_data['search'][0].get('description'),
                'instance_of': self._get_wikidata_values(claims, 'P31'),
                'industry': self._get_wikidata_values(claims, 'P452'),
                'jurisdiction': self._get_wikidata_values(claims, 'P17'),
                'founded': self._get_wikidata_values(claims, 'P571'),
                'website': self._get_wikidata_values(claims, 'P856'),
                'registered_in': self._get_wikidata_values(claims, 'P463')
            }
        except Exception as e:
            print(f"Wikidata query failed: {str(e)}")
            return None

    def _get_wikidata_values(self, claims: Dict, property_id: str) -> List:
        """Extract values from Wikidata claims"""
        if property_id not in claims:
            return []
        
        values = []
        for claim in claims[property_id]:
            if 'datavalue' not in claim.get('mainsnak', {}):
                continue
                
            value = claim['mainsnak']['datavalue']['value']
            if isinstance(value, dict):
                if 'id' in value:
                    values.append(value['id'])
                elif 'time' in value:
                    values.append(value['time'])
                elif 'text' in value:
                    values.append(value['text'])
            else:
                values.append(value)
                
        return values

    def _get_news_data(self, entity_name: str, jurisdiction: str = None) -> Dict:
        """Fetch recent news articles"""
        query = f'"{entity_name}"'
        if jurisdiction:
            query += f' AND "{jurisdiction}"'
        
        params = {
            'q': query,
            'apiKey': self.news_api_key,
            'pageSize': self.max_news_articles,
            'sortBy': 'publishedAt',
            'language': 'en'
        }
        
        try:
            response = requests.get(
                self.news_api,
                params=params,
                timeout=self.request_timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'error':
                print(f"NewsAPI Error: {data.get('message')}")
                return {'articles': []}
                
            return data
        except Exception as e:
            print(f"News API connection error: {e}")
            return {'articles': []}

    def _calc_entity_risk(self, wikidata_info: Optional[Dict]) -> Dict:
        """Calculate risk based on entity type and structure"""
        if not wikidata_info:
            return {
                'score': 60, 
                'weight': 0.25, 
                'factors': ['No Wikidata information available'],
                'description': 'Entity structure unknown'
            }
            
        risk = 20
        reasons = []
        
        # Check for shell company types
        entity_types = wikidata_info.get('instance_of', [])
        if any(t in self.shell_company_types for t in entity_types):
            risk += 50
            reasons.append("Identified as shell company type")
            
        # Check high-risk industries
        industries = wikidata_info.get('industry', [])
        if any(i in self.high_risk_industries for i in industries):
            risk += 30
            reasons.append("High-risk industry")
            
        return {
            'score': min(100, risk),
            'weight': 0.25,
            'factors': reasons or ['Standard corporate structure'],
            'description': 'Entity type and structure risk'
        }

    def _calc_location_risk(self, jurisdiction: Optional[str], wikidata_info: Optional[Dict]) -> Dict:
        """Calculate jurisdiction risk"""
        risk = 20
        reasons = []
        
        # Check explicit jurisdiction
        if jurisdiction and any(j in jurisdiction.lower() for j in self.high_risk_jurisdictions):
            risk += 60
            reasons.append(f"High-risk jurisdiction: {jurisdiction}")
            
        # Check Wikidata country
        if wikidata_info:
            countries = wikidata_info.get('jurisdiction', [])
            if any(c in ['Q5785', 'Q423', 'Q37806'] for c in countries):  # Cayman, Panama, BVI
                risk += 50
                reasons.append("Registered in offshore jurisdiction")
                
        return {
            'score': min(100, risk),
            'weight': 0.25,
            'factors': reasons or ['Standard jurisdiction'],
            'description': 'Jurisdictional risk assessment'
        }

    def _calc_reputation_risk(self, wiki_data: Dict, news_data: Dict) -> Dict:
        """Calculate reputation risk from Wikipedia and news"""
        risk = 20
        reasons = []
        
        # Wikipedia factors
        if wiki_data.get('controversial'):
            risk += 40
            reasons.append("Wikipedia indicates controversy")
            
        # News sentiment analysis
        negative_terms = ['fraud', 'scam', 'launder', 'investigat', 'sue', 'charged']
        negative_articles = sum(
            1 for article in news_data.get('articles', [])
            if any(term in article.get('title', '').lower() or
                  term in article.get('description', '').lower()
                  for term in negative_terms))
        
        if negative_articles:
            risk += min(40, negative_articles * 15)
            reasons.append(f"{negative_articles} negative news articles")
            
        return {
            'score': min(100, risk),
            'weight': 0.3,
            'factors': reasons or ['Clean reputation'],
            'description': 'Reputation and media coverage risk'
        }

    def _calc_financial_risk(self, wikidata_info: Optional[Dict]) -> Dict:
        """Calculate financial transparency risk"""
        if not wikidata_info:
            return {
                'score': 50, 
                'weight': 0.2, 
                'factors': ['No financial data available'],
                'description': 'Financial transparency unknown'
            }
            
        risk = 30
        reasons = []
        
        # Lack of founding date
        if not wikidata_info.get('founded'):
            risk += 20
            reasons.append("No founding date available")
            
        # Lack of official website
        if not wikidata_info.get('website'):
            risk += 15
            reasons.append("No official website")
            
        return {
            'score': min(100, risk),
            'weight': 0.2,
            'factors': reasons or ['Good financial transparency'],
            'description': 'Financial transparency assessment'
        }

    def _calc_confidence(self, wiki_data: Dict, wikidata_info: Optional[Dict], news_data: Dict) -> int:
        """Calculate confidence score (0-100) in assessment"""
        confidence = 0
        max_possible = 0
        
        # Wikipedia confidence (max 30)
        if wiki_data['exists']:
            content_score = 10 if wiki_data.get('extract') else 5
            controversy_score = 10 if wiki_data.get('controversial') else 5
            url_score = 10 if wiki_data.get('url') else 0
            confidence += content_score + controversy_score + url_score
        max_possible += 30
        
        # Wikidata confidence (max 40)
        if wikidata_info:
            base_score = 10
            data_points = sum([
                1 if wikidata_info.get('instance_of') else 0,
                1 if wikidata_info.get('industry') else 0,
                1 if wikidata_info.get('jurisdiction') else 0,
                1 if wikidata_info.get('founded') else 0,
                1 if wikidata_info.get('website') else 0
            ])
            confidence += base_score + (data_points * 6)
        max_possible += 40
        
        # News confidence (max 30)
        if news_data.get('articles'):
            article_count = len(news_data['articles'])
            confidence += 10 + min(20, article_count)
        max_possible += 30
        
        # Normalize to percentage of possible confidence
        if max_possible == 0:
            return 0
        return min(100, int((confidence / max_possible) * 100))

    def _detect_controversy(self, page_data: Dict) -> bool:
        """Detect controversy in Wikipedia data"""
        text = (page_data.get('title', '') + ' ' + page_data.get('extract', '')).lower()
        return ('controvers' in text or 'scandal' in text or
                'disambiguation' in page_data.get('pageprops', {}))

    def _get_risk_level(self, score: float) -> str:
        """Convert score to risk level"""
        if score >= 75: return "High Risk"
        if score >= 50: return "Medium Risk"
        if score >= 25: return "Low Risk"
        return "Very Low Risk"