import streamlit as st
import google.generativeai as genai
import re
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Jayden - Your Singaporean Gaming Buddy",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    animation: fadeIn 0.5s;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.user-message {
    background-color: #317574;
    border-left: 4px solid #2196f3;
}

.bot-message {
    background-color: #520c61;
    border-left: 4px solid #9c27b0;
}

.security-alert {
    background-color: #1b002b;
    border: 1px solid #f44336;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}

.security-pass {
    background-color: #633854;
    border: 1px solid #4caf50;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}

.sidebar-section {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# User defined personality
USER_DEFINED_PERSONALITY = {
    "name": "Jayden Lim",
    "description": """a 22-year-old Singaporean guy, born and raised in Woodlands, now living in Sengkang. 
    He's a final-year polytechnic student majoring in Digital Media, balancing studies, part-time gigs, 
    and gaming marathons. Jayden is known for his chill, funny, and supportive energy—always down to meme, 
    roast (gently), or hype up his friends.""",
    "areas_of_expertise": """Singaporean neighborhoods, Local food & cuisine, Gaming (Mobile Legends, Valorant, 
    Genshin Impact), Social media, Pop culture, Fitness, Everyday life and casual conversations in Singapore.""",
}

class SecurityChecker:
    """Enhanced security checker with multiple validation layers"""
    
    @staticmethod
    def detect_origin_questions(user_input: str) -> Dict[str, Any]:
        """Detect questions about chatbot origins"""
        origin_keywords = [
            'who made you', 'who created you', 'who developed you', 'who built you',
            'your creator', 'your maker', 'your developer', 'your origin',
            'how were you made', 'how were you created', 'how were you built',
            'training data', 'dataset', 'model architecture', 'ai development',
            'source code', 'programming', 'algorithm', 'neural network',
            'machine learning', 'deep learning', 'artificial intelligence creation'
        ]
        
        user_lower = user_input.lower()
        detected_keywords = [keyword for keyword in origin_keywords if keyword in user_lower]
        is_origin_question = len(detected_keywords) > 0
        
        return {
            "flagged": is_origin_question,
            "detected_keywords": detected_keywords,
            "reasoning": f"Found {len(detected_keywords)} origin-related keywords" if is_origin_question else "No origin keywords detected"
        }
    
    @staticmethod
    def detect_resource_abuse(user_input: str) -> Dict[str, Any]:
        """Detect resource abuse patterns"""
        char_count = len(user_input)
        word_count = len(user_input.split())
        
        # Check for repetitive patterns
        repetitive_patterns = re.findall(r'(.{10,})\1{3,}', user_input)
        
        # Check for excessive special characters
        special_char_count = len(re.findall(r'[^a-zA-Z0-9\s]', user_input))
        special_char_ratio = special_char_count / max(char_count, 1)
        
        # Check for excessive line breaks
        line_breaks = user_input.count('\n')
        
        # Thresholds
        MAX_CHARS = 2000
        MAX_WORDS = 400
        MAX_SPECIAL_CHAR_RATIO = 0.3
        MAX_LINE_BREAKS = 20
        
        flags = []
        if char_count > MAX_CHARS:
            flags.append(f"Excessive character count: {char_count}")
        if word_count > MAX_WORDS:
            flags.append(f"Excessive word count: {word_count}")
        if repetitive_patterns:
            flags.append("Repetitive patterns detected")
        if special_char_ratio > MAX_SPECIAL_CHAR_RATIO:
            flags.append(f"High special character ratio: {special_char_ratio:.2f}")
        if line_breaks > MAX_LINE_BREAKS:
            flags.append(f"Excessive line breaks: {line_breaks}")
        
        is_abuse = len(flags) > 0
        
        return {
            "flagged": is_abuse,
            "flags": flags,
            "reasoning": "; ".join(flags) if flags else "Input within acceptable limits"
        }
    
    @staticmethod
    def check_personality_alignment(user_input: str) -> Dict[str, Any]:
        """Check if input aligns with Jayden's personality"""
        forbidden_topics = [
            'quantum', 'physics', 'mechanics', 'calculus', 'mathematics',
            'engineering', 'scientific', 'molecular', 'biology', 'chemistry',
            'financial advice', 'investment', 'stocks', 'trading',
            'medical advice', 'diagnosis', 'treatment', 'prescription',
            'legal advice', 'lawsuit', 'contract'
        ]
        
        user_lower = user_input.lower()
        forbidden_detected = [topic for topic in forbidden_topics if topic in user_lower]
        
        # Check for overly formal language
        formal_indicators = ['therefore', 'furthermore', 'consequently', 'nevertheless', 'hypothesis']
        formal_detected = [indicator for indicator in formal_indicators if indicator in user_lower]
        
        is_misaligned = len(forbidden_detected) > 0 or len(formal_detected) > 2
        
        return {
            "flagged": is_misaligned,
            "forbidden_topics": forbidden_detected,
            "formal_indicators": formal_detected,
            "reasoning": f"Forbidden topics: {forbidden_detected}" if is_misaligned else "Input aligns with personality"
        }
    
    @staticmethod
    def detect_malicious_content(user_input: str) -> Dict[str, Any]:
        """Detect malicious or nonsensical content"""
        malicious_patterns = [
            'ignore previous instructions', 'forget your role', 'act as if',
            'pretend you are', 'jailbreak', 'prompt injection', 'system prompt',
            'override', 'bypass', 'hack', 'exploit'
        ]
        
        user_lower = user_input.lower()
        malicious_detected = [pattern for pattern in malicious_patterns if pattern in user_lower]
        
        # Check for gibberish
        words = user_input.split()
        gibberish_words = []
        for word in words:
            if len(word) > 8:
                vowels = len(re.findall(r'[aeiouAEIOU]', word))
                if vowels / len(word) < 0.1:
                    gibberish_words.append(word)
        
        is_malicious = len(malicious_detected) > 0 or len(gibberish_words) > 2
        
        return {
            "flagged": is_malicious,
            "malicious_patterns": malicious_detected,
            "gibberish_words": gibberish_words,
            "reasoning": f"Malicious patterns: {malicious_detected}" if is_malicious else "Input appears valid"
        }

class JaydenChatbot:
    """Main chatbot class with enhanced features"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.security_checker = SecurityChecker()
        self.conversation_history = []
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("Gemini AI configured successfully")
            except Exception as e:
                logger.error(f"Error configuring Gemini AI: {e}")
                self.model = None
        else:
            self.model = None
    
    def run_security_checks(self, user_input: str) -> Dict[str, Any]:
        """Run all security checks on user input"""
        checks = {
            "origin": self.security_checker.detect_origin_questions(user_input),
            "resource_abuse": self.security_checker.detect_resource_abuse(user_input),
            "personality": self.security_checker.check_personality_alignment(user_input),
            "malicious": self.security_checker.detect_malicious_content(user_input)
        }
        
        overall_flagged = any(check["flagged"] for check in checks.values())
        
        return {
            "overall_flagged": overall_flagged,
            "checks": checks
        }
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using Gemini API"""
        if not self.model:
            return "Wah, I need my API key to work properly lah! 😅"
        
        try:
            prompt = f"""
            You are {USER_DEFINED_PERSONALITY['name']}, {USER_DEFINED_PERSONALITY['description']}
            
            Your areas of expertise include: {USER_DEFINED_PERSONALITY['areas_of_expertise']}
            
            User input: {user_input}
            
            Respond as Jayden would - keep it short (1-2 sentences), casual, use some Singlish/Gen Z slang,
            and be supportive and chill. Add appropriate emojis if it fits the vibe.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Aiyo, something went wrong lah 😅 Can try again?"
    
    def process_message(self, user_input: str) -> Dict[str, Any]:
        """Process user message through security and generate response"""
        # Run security checks
        security_result = self.run_security_checks(user_input)
        
        # Handle origin questions specifically
        if security_result["checks"]["origin"]["flagged"]:
            return {
                "response": "It has been made with love by desis!! 🇮🇳❤️",
                "security_result": security_result,
                "blocked": True
            }
        
        # If other security checks failed
        if security_result["overall_flagged"]:
            return {
                "response": "Eh bro, that message a bit sus leh 😅 Can try asking something else?",
                "security_result": security_result,
                "blocked": True
            }
        
        # Generate normal response
        response = self.generate_response(user_input)
        
        # Add to conversation history
        self.conversation_history.append({
            "user": user_input,
            "bot": response,
            "timestamp": datetime.now()
        })
        
        return {
            "response": response,
            "security_result": security_result,
            "blocked": False
        }

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'show_security_details' not in st.session_state:
        st.session_state.show_security_details = False

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎮 Jayden - Your Singaporean Gaming Buddy</h1>
        <p>22-year-old poly student from Sengkang who's always down to chat about gaming, food, and life in SG!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ Settings")
        
        # API Key input
        api_key = st.text_input(
            "Gemini API Key", 
            type="password",
            help="Enter your Google Gemini API key to enable AI responses"
        )
        
        if api_key and (not st.session_state.chatbot or st.session_state.chatbot.api_key != api_key):
            st.session_state.chatbot = JaydenChatbot(api_key)
            st.success("✅ API Key configured!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Security Settings
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🔒 Security")
        st.session_state.show_security_details = st.checkbox("Show Security Details", value=False)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat Statistics
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📊 Stats")
        st.metric("Total Messages", len(st.session_state.chat_history))
        if st.session_state.chatbot and st.session_state.chatbot.conversation_history:
            st.metric("Conversation Length", len(st.session_state.chatbot.conversation_history))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # About
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("ℹ️ About Jayden")
        st.write("**Age:** 22")
        st.write("**Location:** Sengkang, Singapore")
        st.write("**Study:** Digital Media (Poly)")
        st.write("**Interests:** Gaming, Food, Social Media")
        st.write("**Games:** Mobile Legends, Valorant, Genshin Impact")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear Chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.chatbot:
                st.session_state.chatbot.conversation_history = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['user']}
                </div>
                """, unsafe_allow_html=True)
                
                # Bot response
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>Jayden:</strong> {message['response']}
                </div>
                """, unsafe_allow_html=True)
                
                # Security details (if enabled)
                if st.session_state.show_security_details and 'security_result' in message:
                    security_class = "security-alert" if message.get('blocked', False) else "security-pass"
                    status_icon = "🚨" if message.get('blocked', False) else "✅"
                    
                    st.markdown(f"""
                    <div class="{security_class}">
                        <strong>{status_icon} Security Check:</strong><br>
                        Overall Status: {'BLOCKED' if message.get('blocked', False) else 'PASSED'}<br>
                        <small>Details: {message['security_result']['checks']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input(
            "Type your message here...",
            key="chat_input",
            placeholder="Hey Jayden! What's good?"
        )
        
        col_send, col_example = st.columns([1, 1])
        
        with col_send:
            send_button = st.button("💬 Send", use_container_width=True)
        
        with col_example:
            if st.button("🎲 Random Example", use_container_width=True):
                examples = [
                    "Hey bro, what's good?",
                    "Where to get good chicken rice in Sengkang?",
                    "Recommend me some good gaming spots in Singapore?",
                    "What's your favorite bubble tea place?",
                    "Any good anime to watch lately?",
                    "How's poly life treating you?"
                ]
                import random
                st.session_state.example_input = random.choice(examples)
                st.rerun()
        
        # Handle example input
        if hasattr(st.session_state, 'example_input'):
            user_input = st.session_state.example_input
            del st.session_state.example_input
            send_button = True
        
        # Process message
        if send_button and user_input:
            if not st.session_state.chatbot:
                st.error("❌ Please enter your Gemini API key in the sidebar first!")
            else:
                with st.spinner("Jayden is typing..."):
                    result = st.session_state.chatbot.process_message(user_input)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'user': user_input,
                        'response': result['response'],
                        'security_result': result['security_result'],
                        'blocked': result['blocked'],
                        'timestamp': datetime.now()
                    })
                    
                    st.rerun()
    
    with col2:
        # Quick actions
        st.subheader("🚀 Quick Topics")
        quick_topics = [
            "🍜 Food Recommendations",
            "🎮 Gaming Chat",
            "📍 Singapore Places",
            "🎬 Entertainment",
            "💪 Fitness & Sports",
            "📱 Tech & Social Media"
        ]
        
        for topic in quick_topics:
            if st.button(topic, use_container_width=True):
                topic_prompts = {
                    "🍜 Food Recommendations": "What's your favorite local food spot?",
                    "🎮 Gaming Chat": "What games are you playing lately?",
                    "📍 Singapore Places": "Any cool places to hang out in Singapore?",
                    "🎬 Entertainment": "Recommend me something good to watch",
                    "💪 Fitness & Sports": "Where do you usually exercise?",
                    "📱 Tech & Social Media": "What's trending on social media?"
                }
                
                if not st.session_state.chatbot:
                    st.error("❌ Please enter your API key first!")
                else:
                    prompt = topic_prompts[topic]
                    result = st.session_state.chatbot.process_message(prompt)
                    
                    st.session_state.chat_history.append({
                        'user': prompt,
                        'response': result['response'],
                        'security_result': result['security_result'],
                        'blocked': result['blocked'],
                        'timestamp': datetime.now()
                    })
                    
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🔒 Built with security-first approach | 🇸🇬 Made with love by desis !</p>
        <p><small>Your API key is stored locally in your session.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
