Project Overview
The project, named "Club Wyndham Customer Helper Agent" (TNL-HELP), is a multi-phase initiative aimed at evolving Travel & Leisure's digital channels through Generative AI. The primary goal is to enhance customer experience and reduce the workload on human support teams by providing personalized, 24/7 virtual assistant support.

Scope of Work
Phase Three of the initiative focuses on enhancing an existing Minimum Viable Product (MVP), a Gen AI-powered solution for T+L Customer Center's digital channels. Key enhancements and integrations include:

T+L's Booking API: Integration for booking functionalities.
Live Chat (CX One): Integration for real-time customer interactions.
Analytics: Implementation for performance monitoring and continuous improvement.
Voice Integration: Exploration of voice capabilities.
T+L's Vacation Ownership (VO) club applications (web): Integration with existing club applications.
The team is responsible for designing and developing new capabilities using GenAI technologies, including evaluating and customizing GenAI models, building data pipelines, and designing/developing APIs and applications. This also involves reviewing data source adequacy, performing manual functional testing, and providing recommendations based on regular result reviews.

Expected outputs include:

Voya for VO: Enhanced member profile integration, comprehensive knowledge management, intelligent booking assistance, and reservation change/cancellation capabilities.
Architecture: LLM integration framework with prompt orchestration, guardrails and AI safety implementation (content moderation, PII protection, boundary enforcement, records management), error handling, and performance optimization.
Call Center Taxonomy: Advanced analytics of historical call center transcripts, pattern identification for common member issues, conversation flow optimization, CXOne integration with agent escalation pathways, and performance metrics.
Out of scope for this phase are security/penetration testing, accessibility functionality, UI/UX design, integration with additional T+L products, mobile application integration, and call center agent-facing tools.

Key Challenges and Accomplishments
The team has been actively involved in various critical aspects of the project:

Product Discovery & Requirements Elicitation: Collaborated with stakeholders to refine business needs into actionable functional and non-functional requirements, prioritizing them based on POC vision, business impact, and technical feasibility.
Business Process Mapping: Created detailed end-to-end process maps for current and envisaged workflows in booking, case management, and conversation touchpoints (Chat & Voice), identifying improvement opportunities.
Opportunity Exploration: Conceptualized and documented AI policy guardrails for ethical, regulatory, and organizational alignment of AI systems, implementing guiding principles for accuracy, tone, and consistency in AI-generated responses.
Product Backlog Ownership: Translated business objectives into actionable backlog items focused on AI persona functionality, defining clear acceptance criteria for measurable success.
Solution Architecture and Backend Development: Created solution architecture documents and diagrams, designed the backend of the GenAI solution, analyzed requirements, ran Proof-of-Concepts (PoCs), and proposed technologies. The team implemented key backend components, including real-time voice, tools, integrations, and guardrails.
API Integrations and Prompt Engineering: Defined agentic tools, performed API integrations, and refined chat interactions. The team also developed and maintained Python-based Flask/FastAPI applications for real-time Gen AI solutions, integrating with AWS Bedrock/OpenAI, and designed adapters for seamless data flow with Club Wyndham's existing systems.
RAG Optimization and Prompt Orchestration: Optimized Retrieval-Augmented Generation (RAG) workflow for efficient knowledge base access and integration, and created a prompt orchestration framework for diverse conversation flows.
Testing and Analytics: Implemented an embedded Agent-based testing framework and LLM-based analytics functionality. The team also contributed to scaling the system from a 20% user pilot to full deployment, maintaining performance and reliability.
Technologies used include AWS Bedrock with Claude 3 models (Haiku, Sonnet), AWS Knowledge Base & OpenSearch Serverless, Club Wyndham APIs, TripAuthority API, JIRA, MIRO, GenAI, AWS Services (ECS, Athena), Python, FastAPI, uv, pipecat-ai, OpenAI Realtime API, Gemini Realtime API, LangChain, LangGraph, Langsmith, Github Copilot, and Claud Code.