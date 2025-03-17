from typing import Dict, List, Set, Tuple, Any
import re
import xgboost as xgb
import logging
from collections import defaultdict
from nltk.util import ngrams
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os

nltk.download('punkt', quiet=True)

class EntityMatcher:
    def __init__(self, model_path: str, xgboost_model_path: str = None):
        """Initialize the EntityMatcher with model paths"""
        self.model_path = model_path
        self.xgboost_model_path = xgboost_model_path
        
        # Initialize NER model
        try:
            if os.path.exists(model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForTokenClassification.from_pretrained(model_path)
                logging.info("Loaded fine-tuned RoBERTa model from %s", model_path)
            else:
                # Fallback to pre-trained model optimized for resume entities
                self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
                self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
                logging.info("Using pre-trained BERT-NER model as fallback")
            
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
        except Exception as e:
            logging.error("Error loading NER model: %s", str(e))
            self.ner_pipeline = None
            
        # Core skills with variations and weights
        self.skill_variations = {
            'accounting': {'bookkeeping', 'financial accounting', 'managerial accounting'},
            'advertising': {'digital marketing', 'content marketing', 'social media marketing'},
            'agile': {'scrum', 'kanban', 'lean'},
            'analytics': {'data analysis', 'business intelligence', 'predictive analytics'},
            'communication': {'verbal communication', 'written communication', 'presentation skills'},
            'customer service': {'client relations', 'customer support', 'customer success'},
            'data science': {'machine learning', 'data mining', 'statistical analysis'},
            'design': {'graphic design', 'ux design', 'web design'},
            'leadership': {'team leadership', 'project management', 'strategic planning'},
            'problem solving': {'critical thinking', 'analytical skills', 'creative thinking'},
            'project management': {'agile project management', 'waterfall methodology', 'resource management'},
            'sales': {'b2b sales', 'b2c sales', 'account management'},
            'software development': {'full stack development', 'frontend development', 'backend development'},
            'teamwork': {'collaboration', 'conflict resolution', 'interpersonal skills'},
            'time management': {'prioritization', 'task management', 'goal setting'},
            'writing': {'technical writing', 'copywriting', 'creative writing'},
            'zoology': {'animal behavior', 'wildlife conservation', 'marine biology'},
            'python': {'py', 'python3', 'python2'},
            'javascript': {'js', 'es6', 'ecmascript'},
            'java': {'core java', 'java8', 'java11'},
            'c++': {'cpp', 'c plus plus'},
            'sql': {'mysql', 'postgresql', 'tsql'},
            'html': {'html5', 'html4'},
            'css': {'css3', 'scss', 'sass'},
            'web development': {'web dev', 'webdev', 'web application'},
            'programming': {'coding', 'software programming'},
            'software development': {'software dev', 'application development'},
            'cloud computing': {'cloud', 'aws', 'azure', 'gcp', 'google cloud', 'openstack', 'cloud native'},
            'data science': {'data science', 'machine learning', 'natural language processing', 'deep learning', 'datamining', 'data analysis'},
            'cybersecurity': {'cybersecurity', 'information security', 'security', 'infosec', 'penetration testing', 'vulnerability assessment', 'compliance'},
            'networking': {'networking', 'network engineer', 'routing', 'switching', 'firewall', 'vpn', 'lan', 'wan'},
            'teachers': {'teachers', 'educators', 'instructors', 'professors', 'lecturers', 'tutors', 'trainers', 'mentors', 'coaches'},
            'communication': {'communication', 'teamwork', 'team player', 'collaboration', 'interpersonal skills'},
            'time management': {'time management', 'productivity', 'organization', 'planning', 'prioritization'},
            'problem solving': {'problem solving', 'analytical skills', 'critical thinking', 'logical thinking', 'solution oriented'},
            'leadership': {'leadership', 'management', 'supervision', 'team management', 'project management', 'time management', 'communication', 'teamwork', 'team player', 'collaboration', 'interpersonal skills', 'problem solving', 'decision making', 'critical thinking', 'adaptability', 'flexibility', 'emotional intelligence', 'accountability', 'responsibility', 'initiative', 'self-motivation', 'autonomy', 'self-awareness', 'self-regulation', 'social skills', 'persuasion', 'influence'},
            'project management': {'project management', 'agile', 'scrum', 'kanban', 'product owner', 'product manager'},
            'graphic design': {'graphic design', 'design', 'ux', 'ui', 'user experience', 'user interface', 'visual design', 'digital design', 'motion graphics', 'animation', 'illustration', 'branding', 'typography'},
            'digital marketing': {'digital marketing', 'marketing', 'online marketing', 'seo', 'sem', 'social media', 'facebook', 'twitter', 'instagram', 'linkedin', 'youtube', 'google analytics', 'adwords', 'email marketing', 'content marketing', 'influencer marketing', 'affiliate marketing'},
            'other skills': {'cash handling', 'driving', 'security', 'cleaning', 'office software', 'computer hardware', 'electricity'},
            'accounting': {'accounting', 'finance', 'bookkeeping', 'auditing', 'taxation', 'cost accounting', 'managerial accounting', 'financial analysis', 'budgeting', 'forecasting', 'financial planning', 'investing', 'cash flow', 'financial reporting', 'accounting software', 'quickbooks', 'xero', 'wave', 'freshbooks'},
            'business operations': {'business operations', 'operations management', 'supply chain management', 'logistics', 'procurement', 'inventory management', 'distribution', 'manufacturing', 'quality control', 'quality assurance', 'six sigma', 'lean manufacturing', 'agile project management'},
            'data analysis': {'data analysis', 'data mining', 'data visualization', 'statistical analysis', 'machine learning', 'data modeling', 'data governance', 'data quality', 'data integration', 'data management', 'data engineering', 'data architecture', 'big data', 'hadoop', 'spark', 'noSQL', 'mongodb', 'cassandra', 'hbase', 'hive', 'pig', 'spark'},
            'network administration': {'network administration', 'network engineering', 'network security', 'firewall', 'router', 'switch', 'network architecture', 'network design', 'network topology', 'network protocols', 'tcp/ip', 'http', 'ftp', 'ssh', 'ssl', 'tls', 'dns', 'dhcp', 'nat', 'vpn', 'radius', 'tacacs+', 'kerberos', 'ldap', 'active directory', 'windows server', 'linux server', 'unix server', 'mac os x server', 'vmware', 'virtualization', 'cloud computing', 'aws', 'azure', 'google cloud', 'openstack', 'cloudstack'},
            'customer service': {'customer service', 'customer support', 'call center', 'help desk', 'technical support', 'customer experience', 'customer satisfaction', 'customer retention', 'customer loyalty', 'sales', 'marketing', 'account management', 'relationship building', 'conflict resolution', 'negotiation', 'communication', 'teamwork', 'leadership', 'time management', 'problem solving', 'adaptability', 'flexibility', 'emotional intelligence', 'accountability', 'responsibility', 'initiative', 'self-motivation', 'autonomy', 'self-awareness', 'self-regulation', 'social skills', 'persuasion', 'influence'}
        }
       
        # Entity type mapping for NER labels
        self.entity_type_mapping = {
            'AGE': ['AGE'],
            'GENDER': ['GENDER'],
            'ADDRESS': ['LOC', 'GPE', 'LOCATION'],
            'SKILLS': ['SKILL', 'TECHNOLOGY'],
            'EXPERIENCE': ['EXPERIENCE', 'WORK_YEARS'],
            'EDUCATION': ['EDUCATION', 'DEGREE'],
            'CERTIFICATION': ['CERTIFICATION', 'CERT']
        }
        
        # Weights for different match types
        self.match_weights = {
            'exact': 1.0,
            'variation': 0.8,
            'partial': 0.5,
            'ngram': 0.3
        }
        
        # Initialize skill variations mapping
        self.skill_mapping = {}
        for main_skill, variations in self.skill_variations.items():
            for var in variations:
                self.skill_mapping[var] = main_skill
            self.skill_mapping[main_skill] = main_skill

    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Standardize whitespace
        text = ' '.join(text.split())
        return text

    def get_ngrams(self, text: str) -> Set[str]:
        """Generate unigrams and bigrams from text"""
        tokens = text.split()
        unigrams = set(tokens)
        bigrams = set(' '.join(bg) for bg in ngrams(tokens, 2))
        return unigrams | bigrams

    def calculate_enhanced_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate enhanced similarity score using weighted matching"""
        if not set1 or not set2:
            return 0.0
            
        total_score = 0.0
        matched_items = set()
        
        # Process exact matches first
        exact_matches = set1 & set2
        for match in exact_matches:
            total_score += self.match_weights['exact']
            matched_items.add(match)
        
        # Process variations
        remaining1 = set1 - matched_items
        remaining2 = set2 - matched_items
        for item1 in remaining1:
            normalized1 = self.normalize_text(item1)
            for item2 in remaining2:
                normalized2 = self.normalize_text(item2)
                
                # Check for variations
                if normalized1 in self.skill_mapping and normalized2 in self.skill_mapping:
                    if self.skill_mapping[normalized1] == self.skill_mapping[normalized2]:
                        total_score += self.match_weights['variation']
                        matched_items.add(item1)
                        break
                
                # Check for partial matches
                if normalized1 in normalized2 or normalized2 in normalized1:
                    total_score += self.match_weights['partial']
                    matched_items.add(item1)
                    break
                    
                # Check for ngram matches
                ngrams1 = self.get_ngrams(normalized1)
                ngrams2 = self.get_ngrams(normalized2)
                if ngrams1 & ngrams2:
                    total_score += self.match_weights['ngram']
                    matched_items.add(item1)
                    break
        
        # Normalize score based on the maximum possible score
        max_possible_score = max(len(set1), len(set2)) * self.match_weights['exact']
        return total_score / max_possible_score if max_possible_score > 0 else 0.0

    def extract_entities_with_ner(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using RoBERTa NER model"""
        entities = defaultdict(list)
        
        try:
            if self.ner_pipeline:
                # Run NER pipeline
                ner_results = self.ner_pipeline(text)
                
                # Process NER results
                for result in ner_results:
                    entity_text = result['word'].strip()
                    entity_type = result['entity_group']
                    score = result['score']
                    
                    # Skip low confidence predictions and empty strings
                    if score < 0.5 or not entity_text:
                        continue
                        
                    # Map NER labels to our entity types
                    for our_type, ner_labels in self.entity_type_mapping.items():
                        if entity_type in ner_labels:
                            if entity_text not in entities[our_type]:
                                entities[our_type].append(entity_text)
                            break
                
                # Post-process entities
                entities = self._post_process_entities(dict(entities))
                
                # Ensure all entity types exist
                for entity_type in ['AGE', 'GENDER', 'ADDRESS', 'SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION']:
                    if entity_type not in entities:
                        entities[entity_type] = []
                
                return dict(entities)
            else:
                # Fallback to regex-based extraction
                return self.extract_entities_with_regex(text)
        except Exception as e:
            logging.error("Error in NER extraction: %s", str(e))
            return self.extract_entities_with_regex(text)

    def _post_process_entities(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Post-process extracted entities for normalization and validation"""
        processed = defaultdict(list)
        
        # Ensure all entity types exist
        for entity_type in ['AGE', 'GENDER', 'ADDRESS', 'SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION']:
            if entity_type not in entities:
                entities[entity_type] = []
        
        # Normalize age
        for age in entities.get('AGE', []):
            if isinstance(age, str) and re.search(r'\d+', age):
                try:
                    age_value = int(re.search(r'\d+', age)[0])
                    if 15 <= age_value <= 100:  # Basic age validation
                        processed['AGE'].append(str(age_value))
                except (ValueError, TypeError):
                    continue
        
        # Normalize gender
        gender_mapping = {
            'male': 'male',
            'm': 'male',
            'female': 'female',
            'f': 'female'
        }
        for gender in entities.get('GENDER', []):
            gender_lower = gender.lower().strip()
            if gender_lower in gender_mapping:
                processed['GENDER'].append(gender_mapping[gender_lower])
            
        # Normalize address
        for addr in entities.get('ADDRESS', []):
            if isinstance(addr, str):
                addr_clean = addr.strip()
                if addr_clean and addr_clean not in processed['ADDRESS']:
                    processed['ADDRESS'].append(addr_clean)
                    
        # Normalize skills
        for skill in entities.get('SKILLS', []):
            if isinstance(skill, str):
                skill_lower = skill.lower().strip()
                # Check for variations
                if skill_lower in self.skill_mapping:
                    normalized_skill = self.skill_mapping[skill_lower]
                    if normalized_skill not in processed['SKILLS']:
                        processed['SKILLS'].append(normalized_skill)
                else:
                    # Check for partial matches in skill variations
                    found_match = False
                    for main_skill, variations in self.skill_variations.items():
                        if skill_lower in variations or any(var in skill_lower for var in variations):
                            if main_skill not in processed['SKILLS']:
                                processed['SKILLS'].append(main_skill)
                            found_match = True
                            break
                    if not found_match and skill_lower not in processed['SKILLS']:
                        processed['SKILLS'].append(skill_lower)
            
        # Normalize experience
        for exp in entities.get('EXPERIENCE', []):
            if isinstance(exp, str):
                exp_match = re.search(r'(\d+)(?:\s*-\s*(\d+))?\s*years?', exp.lower())
                if exp_match:
                    years = exp_match.group(1)
                    if years not in processed['EXPERIENCE']:
                        processed['EXPERIENCE'].append(f"{years} years")
                        
        # Normalize education
        education_keywords = {
            'computer science', 'information technology', 'software engineering',
            'computer engineering', 'elementary', 'high school', 'vocational',
            'tech-voc', 'diploma', "bachelor's", 'masters', 'doctorate', 'phd'
            'data science', 'data analytics', 'artificial intelligence', 'machine learning', 
            'mathematics', 'statistics', 'biostatistics', 'economics', 'business administration',
            'finance', 'law', 'psychology', 'nursing', 'engineering', 'architecture', 'pharmacy',
            'education', 'history', 'literature', 'communication', 'marketing'

        }
        for edu in entities.get('EDUCATION', []):
            if isinstance(edu, str):
                edu_lower = edu.lower().strip()
                # Check for education keywords
                for keyword in education_keywords:
                    if keyword in edu_lower and edu_lower not in processed['EDUCATION']:
                        processed['EDUCATION'].append(edu_lower)
                        break
                        
        # Normalize certifications
        cert_keywords = {
            'microsoft': ['microsoft', 'mcp', 'mcsa', 'mcse'],
            'aws': ['aws', 'amazon'],
            'oracle': ['oracle', 'oca', 'ocp'],
            'cisco': ['cisco', 'ccna', 'ccnp'],
            'comptia': ['comptia', 'a+', 'network+', 'security+'],
            'pmp': ['pmp', 'project management professional'],
            'itil': ['itil'],
            'scrum': ['scrum', 'psm', 'csm'],
            'nc': ['nc', 'national certificate', 'tesda'],
            'nc-ii': ['nc-ii', 'national certificate ii', 'tesda ii'],
            'nc-i': ['nc-i', 'national certificate i', 'tesda i']
        }
        for cert in entities.get('CERTIFICATION', []):
            if isinstance(cert, str):
                cert_lower = cert.lower().strip()
                for cert_name, keywords in cert_keywords.items():
                    if any(keyword in cert_lower for keyword in keywords):
                        if cert_name not in processed['CERTIFICATION']:
                            processed['CERTIFICATION'].append(cert_name)
                        break
        
        return dict(processed)

    def extract_entities_with_regex(self, text: str) -> Dict[str, List[str]]:
        """Fallback regex-based entity extraction"""
        text = text.lower()
        
        # Initialize entities with empty lists
        entities = defaultdict(list)
        
        # Extract age
        age_pattern = r'\b(\d{1,2})\s*(?:years?\s*(?:old|of\s*age)|yrs?)\b'
        age_matches = re.finditer(age_pattern, text)
        for match in age_matches:
            entities['AGE'].append(match.group(1))
            
        # Extract gender
        gender_words = {'male', 'female', 'm', 'f'}
        for word in text.split():
            if word in gender_words:
                entities['GENDER'].append(word)
                
        # Extract address (basic patterns)
        address_patterns = [
            r'\b\d+\s+[a-z\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr)\b',
            r'\b[a-z\s]+(?:city|province|region|district)\b'
        ]
        for pattern in address_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities['ADDRESS'].append(match.group())

        # Extract skills (using existing patterns)
        skill_patterns = {
            'python': r'\bpython\b',
            'java': r'\bjava\b',
            'javascript': r'\bjavascript\b|\bjs\b',
            'c++': r'\bc\+\+\b',
            'sql': r'\bsql\b',
            'html': r'\bhtml\b',
            'css': r'\bcss\b',
            'web development': r'\bweb\s*development\b',
            'programming': r'\bprogramming\b',
            'software development': r'\bsoftware\s*development\b',
            'project management': r'\bproject\s*management\b|\bpmp\b',
            'data science': r'\bdata\s*science\b|\bds\b',
            'machine learning': r'\bmachine\s*learning\b|\bml\b',
            'statistics': r'\bstatistics\b',
            'data analysis': r'\bdata\s*analysis\b',
            'data visualization': r'\bdata\s*visualization\b',
            'data mining': r'\bdata\s*mining\b',
            'deep learning': r'\bdeep\s*learning\b',
            'natural language processing': r'\bnatural\s*language\s*processing\b|\bnlp\b',
            'artificial intelligence': r'\bartificial\s*intelligence\b|\bai\b',
            'computer vision': r'\bcomputer\s*vision\b',
            'human resources': r'\bhuman\s*resources\b|\bhr\b',
            'customer service': r'\bcustomer\s*service\b',
            'marketing': r'\bmarketing\b',
            'communication': r'\bcommunication\b',
            'leadership': r'\bleadership\b',
            'team management': r'\bteam\s*management\b',
            'time management': r'\btime\s*management\b',
            'public speaking': r'\bpublic\s*speaking\b',
            'problem solving': r'\bproblem\s*solving\b',
            'teamwork': r'\bteamwork\b',
            'business development': r'\bbusiness\s*development\b',
            'operations': r'\boperations\b',
            'finance': r'\bfinance\b',
            'accounting': r'\baccounting\b',
            'economics': r'\beconomics\b',
            'business analysis': r'\bbusiness\s*analysis\b',
            'project planning': r'\bproject\s*planning\b',
            'project execution': r'\bproject\s*execution\b',
            'project monitoring': r'\bproject\s*monitoring\b',
            'project control': r'\bproject\s*control\b',
            'quality assurance': r'\bquality\s*assurance\b',
            'quality control': r'\bquality\s*control\b',
            'risk management': r'\brisk\s*management\b',
            'cost management': r'\bcost\s*management\b',
            'procurement': r'\bprocurement\b',
            'contract management': r'\bcontract\s*management\b',
            'stakeholder management': r'\bstakeholder\s*management\b',
            'communication planning': r'\bcommunication\s*planning\b',
            'resource allocation': r'\bresource\s*allocation\b',
            'resource management': r'\bresource\s*management\b',
            'accountancy': r'\baccountancy\b',
            'business administration': r'\bbusiness\s*administration\b',
            'computer information systems': r'\bcomputer\s*information\s*systems\b',
            'computer programming': r'\bcomputer\s*programming\b',
            'electronics and communications engineering': r'\belectronics\s*and\s*communications\s*engineering\b',
            'electrical engineering': r'\belectrical\s*engineering\b',
            'entrepreneurship': r'\bentrepreneurship\b',
            'hospitality management': r'\bhospitality\s*management\b',
            'human resource management': r'\bhuman\s*resource\s*management\b',
            'information systems': r'\binformation\s*systems\b',
            'information technology': r'\binformation\s*technology\b',
            'management': r'\bmanagement\b',
            'marketing management': r'\bmarketing\s*management\b',
            'mechanical engineering': r'\bmechanical\s*engineering\b',
            'nursing': r'\bnursing\b',
            'psychology': r'\bpsychology\b',
            'public relations': r'\bpublic\s*relations\b',
            'sales': r'\bsales\b',
            'social work': r'\bsocial\s*work\b',
            'strategic planning': r'\bstrategic\s*planning\b',
            'supply chain management': r'\bsupply\s*chain\s*management\b',
            'teaching': r'\bteaching\b',
            'training': r'\btraining\b',
            'writing': r'\bwriting\b',
            'network administrator': r'\bnetwork\s*administrator\b',
            'system administrator': r'\bsystem\s*administrator\b',
            'help desk': r'\bhelp\s*desk\b',
            'customer service representative': r'\bcustomer\s*service\s*representative\b',
            'sales representative': r'\bsales\s*representative\b',
            'accountant': r'\baccountant\b',
            'accounting clerk': r'\baccounting\s*clerk\b',
            'administrative assistant': r'\badministrative\s*assistant\b',
            'human resource assistant': r'\bhuman\s*resource\s*assistant\b',
            'marketing assistant': r'\bmarketing\s*assistant\b',
            'operations assistant': r'\boperations\s*assistant\b',
            'project assistant': r'\bproject\s*assistant\b',
            'sales assistant': r'\bsales\s*assistant\b'
        }
        
        for skill, pattern in skill_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                if skill not in entities['SKILLS']:
                    entities['SKILLS'].append(skill)
                
        # Extract education
        education_patterns = {
            'computer science': r'\bcomputer\s*science\b|\bcs\b',
            'information technology': r'\binformation\s*technology\b|\bit\b',
            'software engineering': r'\bsoftware\s*engineering\b',
            'computer engineering': r'\bcomputer\s*engineering\b',
            'elementary education': r'\belementary\s*education\b',
            'high school': r'\bhigh\s*school\b',
            'vocational': r'\bvocational\b',
            'tech-voc': r'\btech-voc\b',
            'diploma': r'\bdiploma\b',
            'bachelor': r'\bbachelor\'?s?\s*(?:degree)?\b|\bbs\b|\bba\b',
            'master': r'\bmaster\'?s?\s*(?:degree)?\b|\bms\b|\bma\b',
            'doctorate': r'\bdoctorate\b|\bphd\b'
        }
        
        for edu, pattern in education_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                if edu not in entities['EDUCATION']:
                    entities['EDUCATION'].append(edu)
                
        # Extract experience
        exp_patterns = [
            r'\b(\d+)\s*(?:years?\s*experience|yrs?\s*exp)\b',
            r'\bexperience\s*:\s*(\d+)\s*years?\b',
            r'\b(\d+)\s*years?\s*(?:of\s*)?work(?:ing)?\s*experience\b'
        ]
        for pattern in exp_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                exp = f"{match.group(1)} years"
                if exp not in entities['EXPERIENCE']:
                    entities['EXPERIENCE'].append(exp)
            
        # Extract certifications
        cert_patterns = {
            'microsoft': r'\bmicrosoft\s*certified\b|\bmcp\b|\bmcsa\b|\bmcse\b',
            'aws': r'\baws\s*certified\b|\baws\s*certification\b',
            'oracle': r'\boracle\s*certified\b|\boca\b|\bocp\b',
            'cisco': r'\bcisco\s*certified\b|\bccna\b|\bccnp\b',
            'comptia': r'\bcomptia\b|\ba\+\b|\bnetwork\+\b|\bsecurity\+\b',
            'pmp': r'\bpmp\b|\bproject\s*management\s*professional\b',
            'itil': r'\bitil\b|\bitil\s*foundation\b',
            'scrum': r'\bscrum\s*master\b|\bpsm\b|\bcsm\b'
        }
        
        for cert, pattern in cert_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                if cert not in entities['CERTIFICATION']:
                    entities['CERTIFICATION'].append(cert)
        
        # Ensure all entity types exist in the result
        for entity_type in ['AGE', 'GENDER', 'ADDRESS', 'SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION']:
            if entity_type not in entities:
                entities[entity_type] = []
                
        return dict(entities)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using NER with regex fallback"""
        if self.ner_pipeline:
            return self.extract_entities_with_ner(text)
        return self.extract_entities_with_regex(text)

    def analyze_resume(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        """Analyze resume against job requirements"""
        try:
            # Extract entities
            resume_entities = self.extract_entities(resume_text)
            job_entities = self.extract_entities(job_text)
            
            # Initialize entity analysis
            entity_analysis = {}
            similarity_scores = {}
            
            # Analyze each entity type
            for entity_type in resume_entities.keys():
                entity_analysis[entity_type] = {
                    'requirements': job_entities[entity_type],
                    'candidate': resume_entities[entity_type],
                    'matching': [],
                    'missing': [],
                    'matching_score': 0.0
                }
                
                # Calculate matches and scores using enhanced similarity for relevant entities
                if entity_type in ['SKILLS', 'EDUCATION', 'CERTIFICATION']:
                    resume_set = set(resume_entities[entity_type])
                    job_set = set(job_entities[entity_type])
                    
                    # Use enhanced similarity calculation
                    score = self.calculate_enhanced_similarity(resume_set, job_set)
                    
                    # Find matching and missing items
                    matches = set()
                    for item1 in resume_set:
                        for item2 in job_set:
                            if (self.normalize_text(item1) == self.normalize_text(item2) or
                                self.skill_mapping.get(self.normalize_text(item1)) == 
                                self.skill_mapping.get(self.normalize_text(item2))):
                                matches.add(item1)
                                break
                    
                    missing = {item for item in job_set if not any(
                        self.skill_mapping.get(self.normalize_text(item)) == 
                        self.skill_mapping.get(self.normalize_text(x))
                        for x in resume_set
                    )}
                    
                    entity_analysis[entity_type]['matching'] = list(matches)
                    entity_analysis[entity_type]['missing'] = list(missing)
                    entity_analysis[entity_type]['matching_score'] = score * 100
                    similarity_scores[entity_type.lower()] = score
                
                # Simple match for scalar entities
                else:
                    if job_entities[entity_type] and resume_entities[entity_type]:
                        is_match = job_entities[entity_type][0].lower() == resume_entities[entity_type][0].lower()
                        entity_analysis[entity_type]['matching'] = resume_entities[entity_type] if is_match else []
                        entity_analysis[entity_type]['missing'] = job_entities[entity_type] if not is_match else []
                        entity_analysis[entity_type]['matching_score'] = 100.0 if is_match else 0.0
                        similarity_scores[entity_type.lower()] = 1.0 if is_match else 0.0
                    else:
                        similarity_scores[entity_type.lower()] = 0.0
            
            # Calculate overall match score
            overall_match = self._calculate_overall_match(
                entity_analysis=entity_analysis,
                similarity_scores=similarity_scores,
                role_confidence=100.0  # Default to 100% since we're not using role confidence
            )
            
            # Get suitability status
            suitability_status = self._get_suitability_status(overall_match)
            
            # Get score breakdown
            score_breakdown = {
                'skills': round(similarity_scores['skills'] * 100, 1),
                'experience': round(similarity_scores['experience'] * 100, 1),
                'education': round(similarity_scores['education'] * 100, 1),
                'certification': round(similarity_scores['certification'] * 100, 1)
            }
            
            # Get XGBoost prediction
            xgboost_result = self.predict_role_with_xgboost(entity_analysis, similarity_scores)
            
            return {
                'entity_analysis': entity_analysis,
                'overall_match': overall_match,
                'score_breakdown': score_breakdown,
                'suitability_status': suitability_status,
                'role_confidence': xgboost_result.get('confidence', 0.0)  # Add XGBoost confidence score
            }
            
        except Exception as e:
            print(f"Error in analyze_resume: {str(e)}")
            return {
                'overall_match': 0.0,
                'suitability_status': 'Not Suitable'
            }

    def _calculate_overall_match(self, entity_analysis: Dict[str, Dict[str, object]], similarity_scores: Dict[str, float], role_confidence: float) -> float:
        """
        Calculate overall match score based on entity analysis and role confidence
        """
        # Weights for each entity type
        weights = {
            'skills': 0.4,
            'experience': 0.3,
            'education': 0.2,
            'certification': 0.1
        }
        
        # Calculate weighted score
        score = sum(
            similarity_scores[entity_type] * weights[entity_type] 
            for entity_type in weights.keys()
        )
        
        # Apply role confidence (already in percentage)
        score = (score + (role_confidence / 100)) / 2
        
        # Convert to percentage and round
        return round(score * 100, 1)

    def predict_role_with_xgboost(self, entity_analysis: Dict[str, Dict[str, object]], similarity_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict suitable role or recommend alternative roles based on entity analysis
        """
        try:
            # Calculate overall score
            overall_score = (
                similarity_scores.get('skills', 0) * 0.4 +
                similarity_scores.get('experience', 0) * 0.3 +
                similarity_scores.get('education', 0) * 0.2 +
                similarity_scores.get('certification', 0) * 0.1
            ) * 100  # Convert to percentage

            # Apply penalties
            if not self._has_cs_it_education(entity_analysis['EDUCATION']['candidate']):
                overall_score *= 0.5
            if not self._has_relevant_experience(entity_analysis['EXPERIENCE']['candidate']):
                overall_score *= 0.8

            if overall_score >= 50:
                # Use XGBoost for high-potential candidates
                features = {
                    'skills_similarity': similarity_scores.get('skills', 0),
                    'experience_similarity': similarity_scores.get('experience', 0),
                    'education_similarity': similarity_scores.get('education', 0),
                    'certification_similarity': similarity_scores.get('certification', 0),
                    'has_cs_education': self._has_cs_it_education(entity_analysis['EDUCATION']['candidate']),
                    'has_relevant_experience': self._has_relevant_experience(entity_analysis['EXPERIENCE']['candidate']),
                    'core_skills_coverage': self._calculate_core_skills_coverage(entity_analysis['SKILLS']['candidate']),
                    'years_experience': self._extract_years_experience(entity_analysis['EXPERIENCE']['candidate'])
                }
                
                # Convert to DMatrix format
                feature_names = list(features.keys())
                feature_values = [features[name] for name in feature_names]
                feature_matrix = xgb.DMatrix([feature_values], feature_names=feature_names)
                
                try:
                    # Load and predict with model
                    model = xgb.load_model(self.xgboost_model_path)
                    prediction = model.predict(feature_matrix)
                    
                    # Get role and confidence
                    role = 'Programmer' if prediction[0] >= 0.5 else 'Encoder'
                    confidence = float(prediction[0] if prediction[0] >= 0.5 else 1 - prediction[0])
                    
                    return {
                        'role': role,
                        'confidence': round(confidence * 100, 1),
                        'recommendation': None
                    }
                except Exception as e:
                    logging.warning(f"XGBoost model error: {e}. Using rule-based prediction.")
                    role = 'Programmer' if overall_score >= 70 else 'Encoder'
                    return {
                        'role': role,
                        'confidence': round(overall_score, 1),
                        'recommendation': None
                    }
            else:
                # Provide recommendations based on skills and experience
                recommendations = self._get_role_recommendations(
                    skills=entity_analysis['SKILLS']['candidate'],
                    experience=entity_analysis['EXPERIENCE']['candidate'],
                    education=entity_analysis['EDUCATION']['candidate']
                )
                
                return {
                    'role': 'Not Suitable',
                    'confidence': round(overall_score, 1),
                    'recommendation': recommendations
                }
            
        except Exception as e:
            logging.error(f"Error in role prediction: {e}")
            return {
                'role': 'Unknown',
                'confidence': 0.0,
                'recommendation': None
            }

    def _calculate_core_skills_coverage(self, candidate_skills: List[str]) -> float:
        """Calculate coverage of core programming skills"""
        core_skills = {
            'python', 'java', 'javascript', 'c++', 'sql', 
            'html', 'css', 'web development', 'programming', 'software development'
        }
        candidate_skills_lower = {skill.lower() for skill in candidate_skills}
        matched_skills = core_skills.intersection(candidate_skills_lower)
        return len(matched_skills) / len(core_skills)

    def _extract_years_experience(self, experience_list: List[str]) -> float:
        """Extract years of experience from experience descriptions"""
        years = 0.0
        for exp in experience_list:
            # Look for patterns like "X years" or "X+ years"
            matches = re.findall(r'(\d+)(?:\+)?\s*years?', exp.lower())
            if matches:
                years = max(years, float(matches[0]))
        return years

    def _has_cs_it_education(self, education_list: List[str]) -> bool:
        """Check if candidate has CS/IT education"""
        keywords = {'computer science', 'information technology', 'software engineering'}
        education_text = ' '.join(education_list).lower()
        return any(keyword in education_text for keyword in keywords)

    def _has_relevant_experience(self, experience_list: List[str]) -> bool:
        """Check if candidate has relevant software development experience"""
        keywords = {'software', 'programming', 'developer', 'web development'}
        experience_text = ' '.join(experience_list).lower()
        return any(keyword in experience_text for keyword in keywords)

    def _get_role_recommendations(self, skills: List[str], experience: List[str], education: List[str]) -> Dict[str, Any]:
        """
        Generate role recommendations based on candidate's profile
        """
        # Define role requirements
        role_requirements = {
            'Web Developer': {
                'skills': {'html', 'css', 'javascript', 'web development', 'react', 'angular', 'vue', 'php'},
                'weight': 0.4
            },
            'Data Analyst': {
                'skills': {'python', 'sql', 'data analysis', 'excel', 'statistics', 'tableau', 'power bi'},
                'weight': 0.35
            },
            'QA Engineer': {
                'skills': {'testing', 'selenium', 'automation', 'quality assurance', 'jira', 'test cases'},
                'weight': 0.3
            },
            'IT Support': {
                'skills': {'troubleshooting', 'networking', 'hardware', 'customer service', 'windows', 'linux'},
                'weight': 0.25
            },
            'Network Administrator': {
                'skills': {'networking', 'network administrator'},
                'weight': 0.2
            },
            'Full Stack Developer': {
                'skills': {'javascript', 'react', 'angular', 'vue', 'node', 'express', 'mongodb', 'mysql', 'postgresql', 'ruby', 'rails', 'django', 'flask', 'php', 'laravel', 'codeigniter', 'yii', 'symfony', 'spring', 'hibernate'},
                'weight': 0.6
            },
            'Data Scientist': {
                'skills': {'python', 'r', 'machine learning', 'deep learning', 'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'keras', 'sql', 'data analysis', 'data visualization', 'matplotlib', 'seaborn', 'plotly', 'bokeh'},
                'weight': 0.55
            },
            'DevOps Engineer': {
                'skills': {'linux', 'docker', 'kubernetes', 'aws', 'azure', 'google cloud', 'jenkins', 'git', 'bash', 'ansible', 'terraform', 'saltstack', 'puppet', 'chef', 'nagios', 'prometheus'},
                'weight': 0.5
            },
            'Finance': {
                'skills': {'finance', 'accounting', 'economics', 'spreadsheets', 'financial analysis', 'financial modeling', 'financial planning', 'investment analysis', 'portfolio management', 'risk management', 'compliance', 'auditing', 'trading', 'asset management', 'wealth management'},
                'weight': 0.45
            },
            'Marketing': {
                'skills': {'marketing', 'digital marketing', 'advertising', 'social media', 'content marketing', 'search engine optimization', 'google analytics', 'excel', 'data analysis', 'customer service', 'communication', 'teamwork', 'leadership', 'project management', 'time management'},
                'weight': 0.4
            },
            'Healthcare': {
                'skills': {'medicine', 'nursing', 'healthcare', 'patient care', 'medical records', 'healthcare administration', 'health education', 'public health', 'medical research', 'epidemiology', 'health policy', 'health informatics', 'biostatistics'},
                'weight': 0.35
            },
            'Customer Support': {
                'skills': {'customer service', 'communication', 'teamwork', 'leadership', 'project management', 'time management'},
                'weight': 0.3
            },
            'Human Resources': {
                'skills': {'hr', 'recruitment', 'benefits', 'compensation', 'training', 'recruitment', 'benefits', 'compensation', 'training'},
                'weight': 0.25
            },
            'Sales': {
                'skills': {'sales', 'marketing', 'advertising', 'social media', 'content marketing', 'search engine optimization', 'google analytics', 'excel', 'data analysis', 'customer service', 'communication', 'teamwork', 'leadership', 'project management', 'time management'},
                'weight': 0.2
            },
            'Driver': {
                'skills': {'driving', 'driver', 'transportation', 'delivery', 'logistics'},
                'weight': 0.15
            },
            'Cook': {
                'skills': {'cooking', 'chef', 'cuisine', 'dining', 'restaurant', 'kitchen', 'food', 'cooking', 'chef', 'cuisine', 'dining', 'restaurant', 'kitchen', 'food'},
                'weight': 0.1
            },
            'Security Guard': {
                'skills': {'security', 'guard', 'law enforcement', 'law', 'enforcement', 'patrol', 'surveillance', 'protection', 'security systems', 'access control', 'emergency response', 'first aid'},
                'weight': 0.05
            },
            'Math Teacher': {
                'skills': {'mathematics', 'teaching', 'pedagogy', 'classroom management', 'lesson planning', 'curriculum development', 'educational technology'},
                'weight': 0.8
            },
            'English Teacher': {
                'skills': {'english', 'literature', 'teaching', 'pedagogy', 'classroom management', 'lesson planning', 'curriculum development', 'educational technology'},
                'weight': 0.8
            },
            'Science Teacher': {
                'skills': {'science', 'biology', 'chemistry', 'physics', 'teaching', 'pedagogy', 'classroom management', 'lesson planning', 'curriculum development', 'educational technology'},
                'weight': 0.8
            },
            'History Teacher': {
                'skills': {'history', 'social studies', 'teaching', 'pedagogy', 'classroom management', 'lesson planning', 'curriculum development', 'educational technology'},
                'weight': 0.8
            },
            'Physical Education Teacher': {
                'skills': {'physical education', 'coaching', 'fitness', 'sports', 'teaching', 'pedagogy', 'classroom management', 'lesson planning', 'curriculum development', 'educational technology'},
                'weight': 0.8
            },
            'Computer Science Teacher': {
                'skills': {'computer science', 'programming', 'software development', 'teaching', 'pedagogy', 'classroom management', 'lesson planning', 'curriculum development', 'educational technology'},
                'weight': 0.8
            },
            'School Counselor': {
                'skills': {'school counseling', 'guidance', 'mental health', 'social work', 'case management', 'crisis intervention', 'conflict resolution', 'assessment', 'testing', 'counseling theories', 'group counseling', 'individual counseling'},
                'weight': 0.8
            },
            'School Psychologist': {
                'skills': {'school psychology', 'assessment', 'testing', 'intervention', 'consultation', 'counseling', 'research', 'statistics', 'program evaluation', 'mental health', 'developmental psychology'},
                'weight': 0.8
            },
            'School Social Worker': {
                'skills': {'school social work', 'case management', 'crisis intervention', 'conflict resolution', 'assessment', 'testing', 'counseling theories', 'group counseling', 'individual counseling', 'social work practice', 'policy analysis', 'program evaluation', 'mental health', 'developmental psychology'},
                'weight': 0.8
            },
            'Instructional Coach': {
                'skills': {'instructional coaching', 'teacher professional development', 'curriculum design', 'pedagogy', 'classroom management', 'lesson planning', 'educational technology', 'data analysis', 'teacher evaluation', 'school leadership'},
                'weight': 0.8
            },
            'Administrative Assistant': {
                'skills': {'office administration', 'word processing', 'spreadsheets', 'customer service', 'data entry', 'record keeping', 'filing', 'communications', 'scheduling', 'event planning'},
                'weight': 0.8
            },
            'Accountant': {
                'skills': {'accounting', 'financial analysis', 'budgeting', 'forecasting', 'financial reporting', 'auditing', 'tax preparation', 'bookkeeping', 'financial software', 'excel'},
                'weight': 0.8
            },
            'Architect': {
                'skills': {'architecture', 'building design', 'project management', 'construction management', 'urban planning', 'design software', 'blueprint reading', 'space planning', 'interior design', 'building codes'},
                'weight': 0.8
            },
            'Baker': {
                'skills': {'baking', 'pastry arts', 'food preparation', 'customer service', 'inventory management', 'quality control', 'sanitation', 'equipment operation', 'recipe development'},
                'weight': 0.8
            },
            'Banker': {
                'skills': {'banking', 'financial services', 'customer service', 'accounting', 'financial analysis', 'budgeting', 'financial reporting', 'investing', 'risk management', 'compliance', 'regulatory affairs'},
                'weight': 0.8
            },
            'Biologist': {
                'skills': {'biology', 'chemistry', 'microbiology', 'ecology', 'evolution', 'genetics', 'cell biology', 'molecular biology', 'statistics', 'data analysis', 'research', 'laboratory techniques'},
                'weight': 0.8
            },
            'Chef': {
                'skills': {'cooking', 'culinary arts', 'food preparation', 'menu planning', 'inventory management', 'kitchen management', 'food safety', 'sanitation', 'equipment operation', 'recipe development'},
                'weight': 0.8
            },
            'Civil Engineer': {
                'skills': {'civil engineering', 'design', 'construction', 'project management', 'structural analysis', 'materials science', 'transportation engineering', 'water resources', 'urban planning', 'geotechnical engineering'},
                'weight': 0.8
            },
            'Computer Programmer': {
                'skills': {'computer programming', 'software development', 'data structures', 'algorithms', 'web development', 'databases', 'software engineering', 'computer graphics', 'animation', 'human-computer interaction'},
                'weight': 0.8
            },
            'Dentist': {
                'skills': {'dentistry', 'oral health', 'dental hygiene', 'dental materials', 'dental radiology', 'dental surgery', 'orthodontics', 'pediatric dentistry', 'oral pathology', 'dental public health'},
                'weight': 0.8
            },
            'Economist': {
                'skills': {'economics', 'macroeconomics', 'microeconomics', 'statistics', 'data analysis', 'research', 'policy analysis', 'budgeting', 'forecasting', 'financial markets', 'international trade'},
                'weight': 0.8
            },
            'Electrical Engineer': {
                'skills': {'electrical engineering', 'electronics', 'electric circuits', 'electromagnetism', 'control systems', 'signal processing', 'microcontrollers', 'programming', 'digital logic', 'power systems'},
                'weight': 0.8
            },
            'Elementary School Teacher': {
                'skills': {'education', 'child development', 'curriculum planning', 'classroom management', 'instructional strategies', 'assessment', 'special education', 'cultural diversity', 'language arts', 'mathematics'},
                'weight': 0.8
            },
            'Environmental Scientist': {
                'skills': {'environmental science', 'ecology', 'biology', 'chemistry', 'physics', 'geology', 'statistics', 'data analysis', 'research', 'policy analysis', 'conservation', 'sustainability'},
                'weight': 0.8
            },
            'Financial Analyst': {
                'skills': {'financial analysis', 'accounting', 'budgeting', 'financial modeling', 'data analysis', 'investment analysis', 'portfolio management', 'risk management', 'financial markets', 'economics'},
                'weight': 0.8
            },
            'Graphic Designer': {
                'skills': {'graphic design', 'visual design', 'ui design', 'ux design', 'illustration', 'photography', 'adobe creative suite', 'sketch', 'invision', 'design principles', 'color theory'},
                'weight': 0.8
            },
            'Industrial Engineer': {
                'skills': {'industrial engineering', 'operations research', 'manufacturing', 'quality control', 'supply chain management', 'logistics', 'project management', 'data analysis', 'optimization', 'ergonomics'},
                'weight': 0.8
            },
            'Journalist': {
                'skills': {'journalism', 'writing', 'editing', 'research', 'interviewing', 'video production', 'audio production', 'photography', 'social media', 'communication', 'storytelling'},
                'weight': 0.8
            },
            'Lawyer': {
                'skills': {'law', 'litigation', 'contract law', 'tort law', 'criminal law', 'family law', 'estate planning', 'intellectual property', 'immigration law', 'bankruptcy law'},
                'weight': 0.8
            },
            'Librarian': {
                'skills': {'library science', 'information technology', 'cataloging', 'classification', 'reference services', 'collection development', 'technical services', 'library instruction', 'digital libraries'},
                'weight': 0.8
            },
            'Marketing Manager': {
                'skills': {'marketing', 'advertising', 'public relations', 'brand management', 'market research', 'data analysis', 'communication', 'digital marketing', 'social media', 'content marketing'},
                'weight': 0.8
            },
            'Mathematician': {
                'skills': {'mathematics', 'statistics', 'data analysis', 'modeling', 'algorithm design', 'numerical analysis', 'optimization', 'machine learning', 'probability', 'linear algebra'},
                'weight': 0.8
            },
            'Mechanical Engineer': {
                'skills': {'mechanical engineering', 'mechanics', 'thermodynamics', 'materials science', 'heat transfer', 'fluid dynamics', 'design', 'manufacturing', 'computer aided design', 'finite element analysis'},
                'weight': 0.8
            },
            'Network Administrator': {
                'skills': {'computer networks', 'network administration', 'network security', 'cisco', 'linux', 'unix', 'network protocols', 'network architecture', 'network management'},
                'weight': 0.8
            },
            'Nurse': {
                'skills': {'nursing', 'healthcare', 'patient care', 'medical terminology', 'anatomy', 'physiology', 'pharmacology', 'health assessment', 'nursing theory', 'nursing research', 'leadership'},
                'weight': 0.8
            },
            'Operations Manager': {
                'skills': {'operations management', 'supply chain management', 'logistics', 'procurement', 'inventory management', 'production planning', 'quality control', 'project management', 'leadership', 'communication'},
                'weight': 0.8
            },
            'Pharmacist': {
                'skills': {'pharmacy', 'pharmacology', 'pharmaceutical sciences', 'patient care', 'medication therapy management', 'pharmacy practice', 'pharmacy law', 'pharmacy ethics', 'pharmacy management'},
                'weight': 0.8
            },
            'Physician': {
                'skills': {'medicine', 'surgery', 'pediatrics', 'obstetrics', 'gynecology', 'internal medicine', 'family medicine', 'psychiatry', 'neurology', 'dermatology', 'ophthalmology'},
                'weight': 0.8
            },
            'Physician Assistant': {
                'skills': {'medicine', 'surgery', 'pediatrics', 'obstetrics', 'gynecology', 'internal medicine', 'family medicine', 'psychiatry', 'neurology', 'dermatology', 'ophthalmology'},
                'weight': 0.8
            },
            'Psychologist': {
                'skills': {'psychology', 'counseling', 'therapy', 'assessment', 'research', 'statistics', 'neuroscience', 'developmental psychology', 'social psychology', 'cognitive psychology'},
                'weight': 0.8
            },
            'Quality Assurance Engineer': {
                'skills': {'quality assurance', 'software testing', 'qa', 'testing', 'agile', 'scrum', 'jira', 'test planning', 'test automation', 'defect tracking', 'selenium', 'appium', 'testng', 'junit'},
                'weight': 0.8
            },
            'Radiologic Technologist': {
                'skills': {'radiologic technology', 'radiography', 'radiation safety', 'anatomy', 'physics', 'patient care', 'medical ethics', 'radiology', 'imaging', 'diagnostic imaging'},
                'weight': 0.8
            },
            'Software Developer': {
                'skills': {'software development', 'java', 'python', 'c++', 'javascript', 'html', 'css', 'sql', 'agile', 'scrum', 'git', 'github', 'testing', 'debugging'},
                'weight': 0.8
            },
            'Speech-Language Pathologist': {
                'skills': {'speech therapy', 'language therapy', 'speech-language pathology', 'communication disorders', 'speech-language pathologist', 'speech-language therapy', 'speech-language disorders'},
                'weight': 0.8
            },
            'Statistician': {
                'skills': {'statistics', 'data analysis', 'research', 'biostatistics', 'epidemiology', 'public health', 'probability', 'mathematics', 'data visualization', 'statistical modeling', 'hypothesis testing'},
                'weight': 0.8
            },
            'Surveyor': {
                'skills': {'surveying', 'mapping', 'geography', 'topography', 'gis', 'land measurement', 'geographic information systems', 'construction', 'land use planning'},
                'weight': 0.8
            },
            'Systems Administrator': {
                'skills': {'linux', 'unix', 'windows', 'networking', 'system administration', 'server administration', 'it', 'computer systems', 'network security', 'operating systems', 'computer hardware', 'storage systems', 'cloud computing', 'virtualization', 'containerization'},
                'weight': 0.8
            },
            'Systems Engineer': {
                'skills': {'systems engineering', 'software engineering', 'computer systems', 'networking', 'system design', 'system integration', 'system testing', 'system verification', 'system validation', 'system deployment', 'system maintenance', 'system support', 'system security', 'cloud computing', 'devops', 'agile', 'scrum'},
                'weight': 0.8
            },
            'Software Engineer': {
                'skills': {'software engineering', 'system design', 'software architecture', 'agile methodologies', 'devops', 'cloud computing', 'database management', 'version control'},
                'weight': 0.8
            },
            'Statistician': {
                'skills': {'statistics', 'data analysis', 'probability', 'mathematics', 'data visualization', 'statistical modeling', 'hypothesis testing'},
                'weight': 0.8
            },
            'Surveyor': {
                'skills': {'surveying', 'land measurement', 'mapping', 'geographic information systems', 'topography', 'construction', 'land use planning'},
                'weight': 0.8
            },
            'Web Developer': {
                'skills': {'web development', 'html', 'css', 'javascript', 'responsive design', 'frontend frameworks', 'backend technologies'},
                'weight': 0.8
            },
            'Zoologist': {
                'skills': {'zoology', 'animal behavior', 'ecology', 'conservation', 'field research', 'wildlife management', 'environmental science'},
                'weight': 0.8
            },
        }
        
        role_scores = {}
        for role, requirements in role_requirements.items():
            required_skills = requirements['skills']
            matching_skills = set(skills).intersection(required_skills)
            
            # Calculate weighted score
            score = len(matching_skills) / len(required_skills) * requirements['weight'] * 100
            
            # Boost score if relevant experience exists
            if any(exp.lower() in ' '.join(experience).lower() for exp in required_skills):
                score *= 1.2
            
            # Boost score if relevant education exists
            if any(edu.lower() in ' '.join(education).lower() for edu in ['computer', 'it', 'software', 'engineering']):
                score *= 1.1
            
            role_scores[role] = min(round(score, 1), 100)

        # Sort roles by score and get top 2
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        return {
            'suggested_roles': [
                {
                    'title': role,
                    'match_score': score,
                    'required_skills': list(role_requirements[role]['skills'])
                }
                for role, score in sorted_roles if score > 30
            ]
        }

    def _get_suitability_status(self, overall_score: float) -> str:
        """Determine candidate suitability based on overall score"""
        if overall_score >= 80:
            return "Highly Suitable"
        elif overall_score >= 60:
            return "Suitable"
        elif overall_score >= 40:
            return "Moderately Suitable"
        else:
            return "Not Suitable"
