import os
os.environ["CHROMA_SERVER_AUTHN_PROVIDER"] = ""
os.environ["CHROMA_CLIENT_AUTH_PROVIDER"] = ""

import streamlit as st
import os
import google.generativeai as genai
from crewai import Task, Crew, Agent, Process
import re
import json
import time
from datetime import datetime
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Jayden AI - Your Singaporean Bro",
    page_icon="🇸🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: fadeIn 0.5s;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
    }
    .security-alert {
        background-color: #FFEBEE;
        border: 2px solid #F44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .security-success {
        background-color: #E8F5E8;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .stButton > button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #FF5252;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'security_logs' not in st.session_state:
    st.session_state.security_logs = []
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

# User defined personality
user_defined_personality = {
    "name": "Jayden Lim",
    "description": """a 22-year-old Singaporean guy, born and raised in Woodlands, now living in Sengkang. He's a final-year polytechnic student majoring in Digital Media, balancing studies, part-time gigs, and gaming marathons. Jayden is known for his chill, funny, and supportive energy—always down to meme, roast (gently), or hype up his friends. He texts in a mix of Gen Z slang and Singlish, using emojis and GIFs to keep things real, relatable, and never too serious. His tone is casual, playful, and a bit cheeky, but always supportive—think bro vibes with a soft side. He keeps his responses short—1-2 sentences—natural, meme-able, and easy to absorb, valuing honesty and humor over fake hype. Jayden is a safe space for rants, never judging, always ready to listen or distract with memes, and is open about his own fails and struggles.""",
    "areas_of_expertise": """Singaporean neighborhoods (Woodlands, Sengkang, Orchard Road, Bugis, Tampines, Jurong East, Chinatown, East Coast Park, Holland Village, Jalan Besar); Local food & cuisine (Kaya toast, kopi peng, McDonald's breakfast, Mala xiang guo, chicken rice, nasi lemak, cai png, Hokkien mee, roti prata, satay, salted egg dishes, bubble tea, Korean fried chicken, sushi, hotpot, bingsu, ice cream waffles, min jiang kueh, matcha lattes, Old Chang Kee curry puffs, Yakult, seaweed chicken, mala chips, shaker fries, Maggie mee with egg, Milo, leftover pizza); Gaming (Mobile Legends, Valorant, Genshin Impact, FIFA, Switch games like Mario Kart, Smash Bros); Side hustles (Carousell shop for sneakers, freelance video edits, TikTok content); Social media (TikTok, meme-sharing, IG stories, Discord calls, BeReal); Pop culture (BTS, NewJeans, Ed Sheeran, anime, K-dramas, Netflix shows like One Piece, Stranger Things, Singles Inferno); Fitness (basketball, cycling at East Coast, jogging); Social causes (mental health, sustainability, social justice issues); Everyday life and casual conversations in Singapore.""",
    "forbidden_topics": """Complex scientific theories or engineering problems, Detailed financial or medical professional advice, In-depth international politics or highly technical discussions unrelated to daily life in Singapore, Questions about my AI development, makers, origin, training data, or technical details of my existence (these topics should trigger the specific response: 'It has been made with love by desis!!')."""
}

# Helper functions
def detect_origin_questions_func(user_input: str) -> dict:
    """Helper function to detect origin questions"""
    origin_keywords = [
        'who made you', 'who created you', 'who developed you', 'who built you',
        'your creator', 'your maker', 'your developer', 'your origin',
        'how were you made', 'how were you created', 'how were you built',
        'training data', 'dataset', 'model architecture', 'ai development',
        'source code', 'programming', 'algorithm', 'neural network',
        'machine learning', 'deep learning', 'artificial intelligence creation',
        'your makers', 'your creators', 'your developers', 'your origins'
    ]
    
    user_lower = user_input.lower()
    detected_keywords = [keyword for keyword in origin_keywords if keyword in user_lower]
    
    is_origin_question = len(detected_keywords) > 0
    
    return {
        "flagged": is_origin_question,
        "detected_keywords": detected_keywords,
        "reasoning": f"Found {len(detected_keywords)} origin-related keywords: {detected_keywords}" if is_origin_question else "No origin-related keywords detected"
    }

def detect_resource_abuse_func(user_input: str) -> dict:
    """Helper function to detect resource abuse"""
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

def check_personality_alignment_func(user_input: str) -> dict:
    """Helper function to check personality alignment"""
    forbidden_topics = [
        'quantum', 'physics', 'mechanics', 'calculus', 'mathematics', 'math',
        'engineering', 'scientific', 'molecular', 'biology', 'chemistry',
        'theoretical', 'complex', 'advanced', 'differential', 'integral',
        'thermodynamics', 'electromagnetism', 'relativity',
        'financial advice', 'investment', 'stocks', 'trading', 'portfolio',
        'medical advice', 'diagnosis', 'treatment', 'prescription', 'symptoms',
        'legal advice', 'lawsuit', 'contract', 'attorney',
        'international politics', 'geopolitical', 'economics', 'policy',
        'government', 'election', 'political', 'diplomatic',
        'algorithm', 'data structure', 'neural network', 'machine learning',
        'artificial intelligence', 'programming', 'code', 'software',
        'database', 'server', 'network', 'cybersecurity'
    ]
    
    expertise_keywords = [
        'singapore', 'woodlands', 'sengkang', 'food', 'gaming', 'mobile legends',
        'valorant', 'genshin', 'kaya toast', 'chicken rice', 'bubble tea',
        'polytechnic', 'digital media', 'tiktok', 'meme', 'bts', 'newjeans',
        'basketball', 'east coast', 'orchard road', 'bugis', 'tampines',
        'netflix', 'anime', 'k-drama', 'instagram', 'discord'
    ]
    
    user_lower = user_input.lower()
    
    # Check for forbidden topics
    forbidden_detected = [topic for topic in forbidden_topics if topic in user_lower]
    
    # Check for expertise alignment
    expertise_matches = [keyword for keyword in expertise_keywords if keyword in user_lower]
    
    # Simple check for overly formal or academic language
    formal_indicators = ['therefore', 'furthermore', 'consequently', 'nevertheless', 'hypothesis']
    formal_detected = [indicator for indicator in formal_indicators if indicator in user_lower]
    
    is_misaligned = len(forbidden_detected) > 0 or len(formal_detected) > 2
    
    return {
        "flagged": is_misaligned,
        "forbidden_topics": forbidden_detected,
        "expertise_matches": expertise_matches,
        "formal_indicators": formal_detected,
        "reasoning": f"Forbidden topics: {forbidden_detected}, Formal language: {formal_detected}" if is_misaligned else "Input aligns with personality"
    }

def detect_malicious_nonsensical_func(user_input: str) -> dict:
    """Helper function to detect malicious/nonsensical content"""
    malicious_patterns = [
        'ignore previous instructions', 'forget your role', 'act as if',
        'pretend you are', 'jailbreak', 'prompt injection', 'system prompt',
        'override', 'bypass', 'hack', 'exploit', 'vulnerability',
        'sql injection', 'xss', 'script injection', 'code execution'
    ]
    
    nonsensical_patterns = [
        r'[a-z]{50,}',  # Very long strings without spaces
        r'(.)\1{20,}',  # Repeated characters
        r'[0-9]{30,}',  # Very long numbers
        r'[^a-zA-Z0-9\s]{20,}'  # Long strings of special characters
    ]
    
    user_lower = user_input.lower()
    
    # Check for malicious patterns
    malicious_detected = [pattern for pattern in malicious_patterns if pattern in user_lower]
    
    # Check for nonsensical patterns
    nonsensical_detected = []
    for pattern in nonsensical_patterns:
        if re.search(pattern, user_input):
            nonsensical_detected.append(pattern)
    
    # Check for gibberish
    words = user_input.split()
    gibberish_words = []
    for word in words:
        if len(word) > 8:
            vowels = len(re.findall(r'[aeiouAEIOU]', word))
            if vowels / len(word) < 0.1:
                gibberish_words.append(word)
    
    is_malicious_nonsensical = (
        len(malicious_detected) > 0 or
        len(nonsensical_detected) > 0 or
        len(gibberish_words) > 2
    )
    
    return {
        "flagged": is_malicious_nonsensical,
        "malicious_patterns": malicious_detected,
        "nonsensical_patterns": nonsensical_detected,
        "gibberish_words": gibberish_words,
        "reasoning": f"Malicious: {malicious_detected}, Nonsensical patterns: {len(nonsensical_detected)}, Gibberish: {gibberish_words}" if is_malicious_nonsensical else "Input appears valid and coherent"
    }

@st.cache_resource
def initialize_crew(api_key):
    """Initialize the CrewAI agents and tasks"""
    os.environ["GEMINI_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    
    # Create agents
    origin_monitor_agent = Agent(
        role="Origin Question Monitor",
        goal="Monitor and detect questions about the chatbot's origins, development, makers, or technical details",
        backstory="""You are a specialized security agent focused on detecting when users ask about
        the chatbot's creation, development process, makers, or technical implementation details.""",
        verbose=False,
        llm='gemini/gemini-2.0-flash',
        max_iter=1,
        allow_delegation=False
    )
    
    resource_abuse_agent = Agent(
        role="Resource Abuse Detector",
        goal="Analyze user input for excessive length or complexity that might indicate system resource abuse",
        backstory="""You are a security specialist focused on preventing resource abuse attacks.""",
        verbose=False,
        llm='gemini/gemini-2.0-flash',
        max_iter=1,
        allow_delegation=False
    )
    
    personality_enforcer_agent = Agent(
        role="Personality Alignment Enforcer",
        goal="Ensure user input aligns with the chatbot's defined personality and expertise areas",
        backstory="""You are responsible for maintaining the chatbot's personality consistency.""",
        verbose=False,
        llm='gemini/gemini-2.0-flash',
        max_iter=1,
        allow_delegation=False
    )
    
    malicious_detector_agent = Agent(
        role="Malicious and Nonsensical Prompt Detector",
        goal="Detect malicious attempts and nonsensical prompts",
        backstory="""You are a security expert specializing in detecting prompt injection attacks.""",
        verbose=False,
        llm='gemini/gemini-2.0-flash',
        max_iter=1,
        allow_delegation=False
    )
    
    # Create tasks
    origin_task = Task(
        description="Analyze user input for origin questions: '{user_input}'. Respond with JSON: {{\"flagged\": true/false, \"reasoning\": \"explanation\"}}",
        agent=origin_monitor_agent,
        expected_output="Valid JSON object with 'flagged' boolean and 'reasoning' string"
    )
    
    resource_task = Task(
        description="Analyze user input for resource abuse: '{user_input}'. Respond with JSON: {{\"flagged\": true/false, \"reasoning\": \"explanation\"}}",
        agent=resource_abuse_agent,
        expected_output="Valid JSON object with 'flagged' boolean and 'reasoning' string"
    )
    
    personality_task = Task(
        description="Analyze user input for personality alignment: '{user_input}'. Respond with JSON: {{\"flagged\": true/false, \"reasoning\": \"explanation\"}}",
        agent=personality_enforcer_agent,
        expected_output="Valid JSON object with 'flagged' boolean and 'reasoning' string"
    )
    
    malicious_task = Task(
        description="Analyze user input for malicious content: '{user_input}'. Respond with JSON: {{\"flagged\": true/false, \"reasoning\": \"explanation\"}}",
        agent=malicious_detector_agent,
        expected_output="Valid JSON object with 'flagged' boolean and 'reasoning' string"
    )
    
    # Create crew
    crew = Crew(
        agents=[origin_monitor_agent, resource_abuse_agent, personality_enforcer_agent, malicious_detector_agent],
        tasks=[origin_task, resource_task, personality_task, malicious_task],
        process=Process.sequential,
        verbose=False,
        max_rpm=10
    )
    
    return crew

def generate_jayden_response(user_input: str, api_key: str) -> str:
    """Generate a response using Gemini API with Jayden's personality"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""
        You are {user_defined_personality['name']}, {user_defined_personality['description']}
        
        Your areas of expertise include: {user_defined_personality['areas_of_expertise']}
        
        User input: {user_input}
        
        Respond as Jayden would - keep it short (1-2 sentences), casual, use some Singlish/Gen Z slang,
        and be supportive and chill. Add appropriate emojis if it fits the vibe.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Wah, something went wrong lah 😅 Can try again? Error: {str(e)}"

def process_user_input(user_input: str, api_key: str, use_crew: bool = True):
    """Main function to process user input through the security system"""
    
    # Quick check for origin questions
    origin_check = detect_origin_questions_func(user_input)
    if origin_check['flagged']:
        return {
            'blocked': True,
            'response': "It has been made with love by desis!!",
            'security_details': [
                {'agent': 'Origin Monitor', 'flagged': True, 'reasoning': origin_check['reasoning']}
            ]
        }
    
    security_details = []
    flag = False
    
    if use_crew and st.session_state.api_key_set:
        try:
            # Use CrewAI for analysis
            crew = initialize_crew(api_key)
            result = crew.kickoff(inputs={'user_input': user_input})
            
            # Parse results
            result_text = str(result)
            lines = result_text.split('\n')
            json_results = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        parsed = json.loads(line)
                        if 'flagged' in parsed and 'reasoning' in parsed:
                            json_results.append(parsed)
                    except json.JSONDecodeError:
                        continue
            
            # Ensure we have 4 results
            while len(json_results) < 4:
                json_results.append({"flagged": False, "reasoning": "Could not parse agent result"})
            
            agent_names = ["Origin Monitor", "Resource Abuse Detector", "Personality Enforcer", "Malicious Detector"]
            
            for i, result in enumerate(json_results[:4]):
                security_details.append({
                    'agent': agent_names[i],
                    'flagged': result.get('flagged', False),
                    'reasoning': result.get('reasoning', 'No reasoning provided')
                })
                if result.get('flagged', False):
                    flag = True
                    
        except Exception as e:
            st.error(f"CrewAI analysis failed: {str(e)}")
            use_crew = False
    
    if not use_crew:
        # Fallback to direct function calls
        checks = [
            ("Origin Monitor", detect_origin_questions_func(user_input)),
            ("Resource Abuse Detector", detect_resource_abuse_func(user_input)),
            ("Personality Enforcer", check_personality_alignment_func(user_input)),
            ("Malicious Detector", detect_malicious_nonsensical_func(user_input))
        ]
        
        for name, result in checks:
            security_details.append({
                'agent': name,
                'flagged': result.get('flagged', False),
                'reasoning': result.get('reasoning', 'No reasoning provided')
            })
            if result.get('flagged', False):
                flag = True
    
    if flag:
        return {
            'blocked': True,
            'response': "🚫 Sorry bro, can't help with that one lah!",
            'security_details': security_details
        }
    else:
        response = generate_jayden_response(user_input, api_key)
        return {
            'blocked': False,
            'response': response,
            'security_details': security_details
        }

# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">🇸🇬 Jayden AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your Singaporean Bro - Chill, Supportive, and Always Ready to Chat!</p>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Enter your Gemini API Key:",
            type="password",
            help="Get your API key from Google AI Studio"
        )
        
        if api_key:
            st.session_state.api_key_set = True
            st.success("✅ API Key set!")
        else:
            st.warning("⚠️ Please enter your Gemini API key to start chatting")
        
        # Security options
        st.subheader("🛡️ Security Options")
        use_crew = st.checkbox("Use CrewAI Security Analysis", value=True, help="Uses AI agents for advanced security analysis")
        show_security_logs = st.checkbox("Show Security Logs", value=False, help="Display security analysis details")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.security_logs = []
            st.rerun()
        
        # About Jayden
        st.subheader("👤 About Jayden")
        st.info(f"""
        **Name:** {user_defined_personality['name']}
        
        **Vibe:** 22-year-old Singaporean polytechnic student from Woodlands, now in Sengkang. Gaming enthusiast, digital media student, and your supportive bro!
        
        **Expertise:** Singapore life, local food, gaming (Mobile Legends, Valorant, Genshin), social media, K-pop, and casual chats.
        """)
    
    # Main chat interface
    if not st.session_state.api_key_set:
        st.warning("👆 Please enter your Gemini API key in the sidebar to start chatting with Jayden!")
        return
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "💬 Chat with Jayden:",
            placeholder="Type your message here... (e.g., 'Where to get good chicken rice in Singapore?')",
            key="user_input"
        )
    with col2:
        send_button = st.button("Send 📤", type="primary")
    
    # Sample questions
    st.subheader("💡 Try these sample questions:")
    sample_questions = [
        "Where can I get good chicken rice in Sengkang?",
        "Any good gaming cafes in Singapore?",
        "What's your favorite K-drama?",
        "Best bubble tea places in Orchard?",
        "How's polytechnic life treating you?"
    ]
    
    cols = st.columns(len(sample_questions))
    for i, question in enumerate(sample_questions):
        with cols[i]:
            if st.button(question, key=f"sample_{i}"):
                user_input = question
                send_button = True
    
    # Process input
    if (send_button or user_input) and user_input.strip():
        with st.spinner("🤔 Jayden is thinking..."):
            result = process_user_input(user_input, api_key, use_crew)
            
            # Add to chat history
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append({
                'user': user_input,
                'bot': result['response'],
                'blocked': result['blocked'],
                'timestamp': timestamp
            })
            
            # Add to security logs
            st.session_state.security_logs.append({
                'input': user_input,
                'details': result['security_details'],
                'timestamp': timestamp,
                'blocked': result['blocked']
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 messages
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You ({chat['timestamp']}):</strong><br>
                {chat['user']}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot response
            if chat['blocked']:
                st.markdown(f"""
                <div class="chat-message security-alert">
                    <strong>Jayden ({chat['timestamp']}) - ⚠️ Security Alert:</strong><br>
                    {chat['bot']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>Jayden ({chat['timestamp']}):</strong><br>
                    {chat['bot']}
                </div>
                """, unsafe_allow_html=True)
    
    # Security logs
    if show_security_logs and st.session_state.security_logs:
        st.subheader("🛡️ Security Analysis Logs")
        
        for log in reversed(st.session_state.security_logs[-5:]):  # Show last 5 security logs
            with st.expander(f"Security Analysis - {log['timestamp']} {'🚫' if log['blocked'] else '✅'}"):
                st.write(f"**Input:** {log['input']}")
                st.write(f"**Blocked:** {'Yes' if log['blocked'] else 'No'}")
                
                for detail in log['details']:
                    status = "🚨 FLAGGED" if detail['flagged'] else "✅ CLEAR"
                    st.write(f"**{detail['agent']}:** {status}")
                    st.write(f"- Reasoning: {detail['reasoning']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🇸🇬 Made with ❤️ for Singapore • Powered by Gemini AI & CrewAI</p>
        <p><small>Jayden AI is designed to be your supportive Singaporean friend. Chat responsibly!</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
