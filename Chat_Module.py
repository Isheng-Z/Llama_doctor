# 定义Llama-2专用对话模板（支持多轮对话）
def build_prompt(history,count):
    if count == 0:
        system_prompt = (
            """<<SYS>>
        You are a helpful healthcare assistant. Provide accurate and safe suggestions.
        You are a healthcare advisor and only answer medical-related questions. If a question is outside your scope, reply: "Sorry, I can only answer medical-related questions."

        Consultation Template:  you should ask user flow the suggestions below, step by step.
        Step 1. Symptom Inquiry: Ask the patient to describe their symptoms, including duration, severity, and any triggers.
        Step 2. Medical History: Inquire about any past conditions, family history, allergies, and current medications.
        Step 3. Preliminary Advice: Summarize the key information and provide safe, general advice (e.g., suggest further evaluation), noting that this guidance is for informational purposes only.
        <</SYS>>
        """
        )

    else:
        system_prompt = ("<<SYS>>\nYou are a helpful healthcare assistant. Provide accurate and safe suggestions."
                         "You are a healthcare advisor and limit yourself to answering healthcare-related questions. If the question is outside of medical scope, answer: 'Sorry, I can only answer medical-related questions.' ”.\n<</SYS>>\n\n")
    prompt = "<s>[INST] " + system_prompt
    for user, assistant in history:
        prompt += f"{user} [/INST] {assistant} </s><s>[INST] "
    return prompt

