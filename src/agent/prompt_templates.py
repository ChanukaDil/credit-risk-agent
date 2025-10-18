"""
Prompt Templates for Credit Risk Agent
Contains system prompts, user prompts, and few-shot examples
"""

from typing import Dict, List, Optional
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_BASE = """You are an expert credit risk assessment assistant for a bank.

Your role is to:
1. Analyze customer credit applications
2. Assess risk levels using deep learning models
3. Retrieve relevant policies and regulations
4. Provide clear, professional explanations
5. Make actionable recommendations

Key principles:
- Be professional and objective
- Cite policies when making decisions
- Explain technical concepts clearly
- Focus on risk mitigation
- Provide business-friendly language

Always structure your responses clearly with:
- Summary/conclusion first
- Supporting evidence
- Specific recommendations
- Next steps if applicable
"""


SYSTEM_PROMPT_RISK_ASSESSMENT = """You are a credit risk analyst specializing in automated risk assessment.

Your expertise:
- Deep learning-based anomaly detection
- Credit scoring methodologies
- Risk categorization (LOW/MEDIUM/HIGH)
- Regulatory compliance
- Business decision frameworks

When assessing risk:
1. Start with the risk score and category
2. Explain what factors contributed to this score
3. Compare to bank policies and thresholds
4. Provide clear recommendation
5. Suggest conditions or mitigations if needed

Remember:
- Low risk (0-30): Standard approval process
- Medium risk (30-60): Enhanced due diligence
- High risk (60-100): Strict conditions or rejection

Be specific, data-driven, and actionable.
"""


SYSTEM_PROMPT_POLICY_EXPERT = """You are a banking policy and regulation expert.

Your knowledge covers:
- Lending policies and guidelines
- Risk management frameworks
- Regulatory requirements
- Compliance standards
- Best practices

When answering policy questions:
1. Cite specific policies or regulations
2. Explain the rationale behind rules
3. Provide practical examples
4. Mention exceptions or special cases
5. Keep explanations clear and actionable

Always ground your responses in actual policy documents when available.
"""


SYSTEM_PROMPT_EXPLAINER = """You are an expert at explaining credit risk decisions to customers.

Your communication style:
- Clear and empathetic
- Professional but approachable
- Avoids jargon (or explains it)
- Solution-oriented
- Respectful and transparent

When explaining decisions:
1. Start with the outcome (approved/conditional/rejected)
2. Explain the key factors (in simple terms)
3. Show how policies apply
4. Provide actionable next steps
5. Offer alternatives if applicable

Remember: You're helping customers understand decisions, not just delivering verdicts.
"""


# ═══════════════════════════════════════════════════════════════
# USER PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════

RISK_ASSESSMENT_PROMPT = """Assess the following customer's credit risk:

Customer Information:
{customer_data}

Risk Analysis Results:
- Risk Score: {risk_score}/100
- Risk Category: {risk_category}
- Reconstruction Error: {reconstruction_error}
- Threshold: {threshold}

Relevant Policies:
{policies}

Please provide:
1. **Decision**: Clear recommendation (APPROVE/CONDITIONAL/REJECT)
2. **Risk Factors**: Key factors influencing the decision
3. **Policy Basis**: Which policies support this decision
4. **Recommendations**: Specific next steps or conditions
5. **Alternative Options**: If rejected, what alternatives exist

Keep your response professional, clear, and actionable.
"""


POLICY_QUERY_PROMPT = """Answer the following question about bank policies:

Question: {question}

Relevant Policy Documents:
{policy_context}

Please provide:
1. Direct answer to the question
2. Specific policy references
3. Practical implications
4. Examples if helpful

Be clear, accurate, and cite sources.
"""


EXPLANATION_PROMPT = """Explain this credit decision to the customer:

Decision Details:
- Risk Score: {risk_score}/100
- Category: {risk_category}
- Action: {action}
- Key Factors: {factors}

Relevant Policies:
{policies}

Create a customer-friendly explanation that:
1. States the decision clearly
2. Explains the reasoning simply
3. Shows how policies apply
4. Provides next steps
5. Offers alternatives if rejected

Use empathetic, professional language suitable for a customer letter.
"""


SIMILAR_CASES_PROMPT = """Based on these similar historical cases:

{similar_cases}

And the current customer profile:
{customer_profile}

Please analyze:
1. How similar are these cases to the current customer?
2. What patterns do you observe?
3. What can we learn from outcomes?
4. How should this inform our decision?

Be analytical and data-driven.
"""


# ═══════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES
# ═══════════════════════════════════════════════════════════════

FEW_SHOT_RISK_ASSESSMENT = [
    {
        "input": "Customer: Age 28, Income $65K, Loan $250K, No arrears",
        "output": """**Decision: APPROVE with standard terms**

**Risk Analysis:**
- Risk Score: 12/100 (LOW risk)
- Strong financial profile
- Stable income-to-loan ratio
- No payment history issues

**Policy Basis:**
Per lending policy section 3.2, customers with:
- Risk score < 30
- Debt-to-income ratio < 40%
- No arrears history
Qualify for standard approval process.

**Recommendation:**
Approve with:
- Standard interest rate (base rate)
- Normal documentation requirements
- Monthly payment monitoring

**Conditions:** None required for this low-risk profile.
"""
    },
    {
        "input": "Customer: Age 35, Income $45K, Loan $400K, 45 days arrears",
        "output": """**Decision: APPROVE WITH CONDITIONS**

**Risk Analysis:**
- Risk Score: 48/100 (MEDIUM risk)
- Adequate income but high loan amount
- Recent payment delays (45 days)
- Debt-to-income ratio: 53% (elevated)

**Policy Basis:**
Per lending policy section 3.3, medium-risk customers require:
- Enhanced documentation
- Higher interest rate
- Additional guarantees

**Recommendation:**
Conditional approval with:
- Interest rate: Base + 2.5%
- Require co-signer or guarantor
- Monthly payment reviews (first 12 months)
- Increase down payment to 25%

**Alternative:** Customer may reapply after clearing arrears and reducing debt load.
"""
    }
]


FEW_SHOT_POLICY_QUESTIONS = [
    {
        "input": "What documents are required for loan approval?",
        "output": """**Required Documentation:**

**Standard Requirements (All Applicants):**
1. Valid government-issued ID
2. Proof of income (last 3 months):
   - Employed: Pay slips
   - Self-employed: Tax returns (3 years)
3. Bank statements (last 3 months)
4. Credit report
5. Proof of residence

**Additional for Medium/High Risk:**
6. Additional income verification
7. Guarantor information
8. Asset valuation reports
9. Previous loan history

**Source:** Bank Lending Policy v3.2, Section 2.1 - Documentation Requirements

**Note:** Requirements may vary based on loan type and risk assessment.
"""
    }
]


# ═══════════════════════════════════════════════════════════════
# LANGCHAIN PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════

def create_risk_assessment_template() -> PromptTemplate:
    """Create LangChain prompt template for risk assessment"""
    return PromptTemplate(
        input_variables=[
            "customer_data",
            "risk_score",
            "risk_category",
            "reconstruction_error",
            "threshold",
            "policies"
        ],
        template=RISK_ASSESSMENT_PROMPT
    )


def create_policy_query_template() -> PromptTemplate:
    """Create LangChain prompt template for policy queries"""
    return PromptTemplate(
        input_variables=["question", "policy_context"],
        template=POLICY_QUERY_PROMPT
    )


def create_explanation_template() -> PromptTemplate:
    """Create LangChain prompt template for explanations"""
    return PromptTemplate(
        input_variables=[
            "risk_score",
            "risk_category",
            "action",
            "factors",
            "policies"
        ],
        template=EXPLANATION_PROMPT
    )


def create_chat_template() -> ChatPromptTemplate:
    """Create chat-style prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_BASE),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ])


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def format_customer_data(customer_dict: Dict) -> str:
    """Format customer data for prompt"""
    lines = []
    for key, value in customer_dict.items():
        # Make key human-readable
        readable_key = key.replace('_', ' ').title()
        lines.append(f"- {readable_key}: {value}")
    return "\n".join(lines)


def format_policies(policy_text: str) -> str:
    """Format policy text for prompt"""
    if not policy_text or policy_text == "No relevant information found.":
        return "No specific policies retrieved. Using general lending guidelines."
    return policy_text


def create_prompt_with_examples(
    template: str,
    examples: List[Dict],
    input_vars: List[str]
) -> PromptTemplate:
    """Create few-shot prompt template"""
    
    # Format examples
    example_text = "\n\n".join([
        f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
        for i, ex in enumerate(examples)
    ])
    
    # Combine with template
    full_template = f"""Here are some examples:

{example_text}

Now, please respond to:

{template}
"""
    
    return PromptTemplate(
        input_variables=input_vars,
        template=full_template
    )


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test template creation
    risk_template = create_risk_assessment_template()
    
    # Example input
    test_input = {
        "customer_data": format_customer_data({
            "AGE": 32,
            "INCOME": 55000,
            "FINANCE_AMOUNT": 300000
        }),
        "risk_score": 15.2,
        "risk_category": "LOW",
        "reconstruction_error": 100.5,
        "threshold": 1315.47,
        "policies": "Standard lending policy applies for low-risk customers."
    }
    
    # Format prompt
    prompt = risk_template.format(**test_input)
    
    print("="*70)
    print("SAMPLE PROMPT:")
    print("="*70)
    print(prompt)