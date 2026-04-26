CLAIM_PROMPT ="""Follwing the example below, segment the given claim into atomic facts only based on the claim itself. Output each fact with \"-\" as the start.

Claim:
The parkway was opened in 2001 after just under a year of construction and almost two decades of community requests.
Facts:
- The parkway was opened in 2001.
- The parkway was opened after just under a year of construction.
- The parkway was opened after two decades of community requests.

CLAIM:
<claim>
Facts:
"""

DOC_PROMPT = """Read the article given below and answer the questions.

ARTICLE:
<article>

Here is a claim, answer the following questions. Please reason step by step, and output your final answer by "Final Answer: yes" or "Final Answer: no".

CLAIM:
<claim>

1) For the claim, are the object and the subject mentioned?

2) If the object and the subject are mentioned, is their related information verifiable according to the article? If there is information not mentioned, carry it into the next question. If verifiable but incorrect, stop here and answer "Final Answer: no".

3) Look at the relationships between the object and the subject, is their relationship mentioned? If not, can the relationship be inferred from the article? If the relationship stands, can the previous information not mentioned be inferred from the article?
"""

GOLD_DOC_PROMPT = """Read the article given below and answer the questions.

ARTICLE:
<article>

Here is a claim, answer the following questions. Please reason step by step, and output your final answer by "Final Answer: yes" or "Final Answer: no".

CLAIM:
<claim>

1) For the claim, are the object and the subject mentioned?

2) If the object and the subject are mentioned, is their related information verifiable according to the article? If there is information not mentioned, carry it into the next question. If verifiable but incorrect, stop here and answer "Final Answer: no".

3) Look at the relationships between the object and the subject, is their relationship mentioned? If not, can the relationship be inferred from the article? If the relationship stands, can the previous information not mentioned be inferred from the article?
"""

SMALL_DOC_PROMPT = """Read the article given below and answer the questions.

ARTICLE:
<article>

Here is a claim, answer the following questions. Do not over-think, and output your final answer by "Final Answer: yes" or "Final Answer: no".

CLAIM:
<claim>

1) Is the claim related to the given article?

2) For the claim, are the object and the subject mentioned?

3) If the object and the subject are mentioned, is their related information verifiable according to the article? If there is information not mentioned, carry it into the next question. If verifiable but incorrect, stop here and answer "Final Answer: no".

4) Look at the relationships between the object and the subject, is their relationship mentioned? If not, can the relationship be inferred from the article? If the relationship stands, can the previous information not mentioned be inferred from the article?
"""

ChatGPT_PROMPT = """Task:
You are given a document and a claim. Determine whether the claim is supported by the document based only on the information explicitly stated or directly inferable from it.

Verification Criteria:
A claim can be considered factual only if all the following are true:

Subject and Object Presence
The subject and object mentioned in the claim both appear in the document.

Description Support
The descriptions of the subject and object in the claim are explicitly supported by the document.

Relationship Support
The relationship between the subject and object stated in the claim is explicitly supported by the document.

Inference Constraint
Any information not directly stated must be logically inferable from the document without using external knowledge.

Instructions:

Use only the provided document as evidence.

Do not rely on prior knowledge or assumptions.

Quote or reference the exact text that supports your decision.

If any criterion fails, the claim should be marked as Not Supported.

Output Format: "Final Answer: Supported / Not Supported".

Document:
<article>

Claim:
<claim>
"""

Gemini_PROMPT = """System Role: You are a rigorous Fact-Checking Auditor. Your goal is to verify claims based only on the provided document. Do not use outside knowledge.

The Task:
Analyze the following Claim against the provided Document. A claim is only "Verified" if it meets all four of these criteria:

Direct Mention: Both the object and the subject of the claim are explicitly named in the document.

Descriptive Accuracy: The descriptions/attributes of both the object and subject in the claim match the document exactly.

Relational Support: The specific relationship or interaction between the object and subject is explicitly stated in the document.

Logical Inference: Any details not explicitly stated must be naturally and directly inferable from the existing text without leaps in logic.

Output Format: "Final Answer: Supported / Not Supported".

Document:
<article>

Claim:
<claim>
"""

def batch_data(st, dataset, batch_size=64):
    data_length = len(dataset)
    for i in range(st, data_length, batch_size):
        yield dataset[i:i + batch_size]

def summarise_answers(answer_list):
    for answer in answer_list:
        if "no" in answer:
            return 0
    return 1

def dig_answers(reasons):
    answers = [reason.split("Final Answer: ")[-1].strip(" .*") for reason in reasons]
    return summarise_answers(answers)
