import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_path = "./llama3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=1)

#Extract text from the book's computer networking
large_texts = []
for i in range(1, 10):
    filename_1 = f"Chapter {i}.1.txt"
    filename_2 = f"Chapter {i}.2.txt"    
    chapter_text = ""
    for filename in [filename_1, filename_2]:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
                chapter_text += content + "\n"  # Add the content of both subchapters
        else:
            print(f"File {filename} not found!")
    
    if chapter_text:
        large_texts.append(chapter_text)


###############################################################
# Agent1:Analyze the text, extract the teaching objectives and segment them, and output <END> at the end.
###############################################################
class MaterialAnalysisAndSegmentationAgent:
    def __init__(self, name="MaterialAnalysisAndSegmentationAgent"):
        self.name = name

    def analyze_and_segment(self, text):
        prompt = (
            "You are an AI assistant that analyzes an educational text.\n"
            "Tasks:\n"
            "1. Divide the text into segments, each corresponding to a central content.(Each segment is a piece of the original text)\n"
            "2. Infer the teaching objectives from the segments.\n"
            "3. For each segment, extract key concepts/keywords.\n\n"
        
            "Attention:\n"
            "1. Do not use unformatted punctuation marks such as '*' in your output."
            "2. The output of the sample should not appear in the output."
            "3. After output, output <END> on a new line.\n"
            
            "Output Format:\n"
            "Number of Objectives: n\n"
            "Objective i: the extracted objective\n"
            "Segment i:\n the text segment\n"
            "Keywords i:\n comma-separated keywords\n"
            "<END>\n\n"
            
            "Example Input:\n"
            "An early cipher (circa 45 BC, when we were all little kids). Key in the range (1 .. 25). A → C; B → D; C → E; … Z → B etc when key = 2. A → E; B → F; C → G; … Z → D etc when key = 4. y = (x + key) mod 26."
            "Examples: YOU → AQW (key = 2); YOU → CSY (key = 4). A = 65, the ASCII value of the character A. Y = 89, the ASCII value of the character Y. Assume key = 4. Y then becomes 65 + (((89-65)+4) mod 26). 89-65 = 24; "
            "This says Y is the 24th character in the alphabet. Add 4 (mod 26): you get 2 - i.e., the 2nd character in the alphabet: C (A being the 0th character). To get the ASCII value of C, add 65 to 2. C = 67.\n"
            
            "Example Output:\n"
            "Number of Objectives: 2\n"
            "Objective 1: Understand the Caesar Cipher encryption method\n"
            "Segment 1:\n• An early cipher (circa 45 BC, when we were all little kids).\n"
            "• Key in the range (1 .. 25)\n"
            "• A → C; B → D; C → E; … Z → B etc when key = 2\n"
            "• A → E; B → F; C → G; … Z → D etc when key = 4\n"
            "• y = (x + key) mod 26\n"
            "Examples: YOU → AQW (key = 2); YOU → CSY (key = 4)\n"
            "Keywords 1:\nCaesar Cipher, encryption, key, mod 26, examples\n"
            "Objective 2: Learn how to implement the Caesar Cipher\n"
            "Segment 2:\n• A = 65, the ASCII value of the character A.\n"
            "• Y = 89, the ASCII value of the character Y.\n"
            "• Assume key = 4\n"
            "• Y then becomes 65 + (((89-65)+4) mod 26)\n"
            "• 89-65 = 24; This says Y is the 24th character in the alphabet\n"
            "• Add 4 (mod 26): you get 2 - i.e., the 2nd character in the alphabet: C (A being the 0th character).\n"
            "• To get the ASCII value of C, add 65 to 2. C = 67.\n"
            "Keywords 2:\nimplementation, ASCII, key, Caesar Cipher, encryption, decryption\n"
            "<END>"
        )

        full_prompt = prompt + f"Text:\n\"{text}\"\n\n"
        response = pipe(full_prompt, max_new_tokens=1500, return_full_text=False)[0]['generated_text']
        return self.parse_response(response)

    def parse_response(self, text):
        if "<END>" in text:
            text = text.split("<END>")[0].strip()
        lines = text.strip().split('\n')
        n = 0
        objectives = []
        segments = []
        keywords_list = []
        current_obj = None
        current_seg = []
        current_keys = []
        parsing_seg = False
        parsing_key = False

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower().startswith("number of objectives:"):
                parts = line_stripped.split(':',1)
                if len(parts)>1:
                    try:
                        n = int(parts[1].strip())
                    except:
                        n = 0
            elif line_stripped.lower().startswith("objective"):
                if current_obj:
                    objectives.append(current_obj)
                    segments.append('\n'.join(current_seg))
                    keywords_list.append(current_keys)
                current_seg = []
                current_keys = []
                parsing_seg = False
                parsing_key = False
                parts = line_stripped.split(':',1)
                if len(parts)>1:
                    current_obj = parts[1].strip()
                else:
                    current_obj = ""
            elif line_stripped.lower().startswith("segment"):
                try:
                    line_stripped = line_stripped.split(':', 1)[1]
                except IndexError:
                    line_stripped = ""
                parsing_seg = True
                parsing_key = False
                current_seg = []
            elif line_stripped.lower().startswith("keywords"):
                try:
                    line_stripped = line_stripped.split(':',1)[1]
                except IndexError:
                    line_stripped = ""
                parsing_seg = False
                parsing_key = True
                current_keys = []
            if parsing_seg or parsing_key:
                if parsing_seg:
                    current_seg.append(line_stripped)
                elif parsing_key:
                    key_str = line_stripped
                    for k in key_str.split(','):
                        k = k.strip()
                        if k:
                            current_keys.append(k)
                            

        if current_obj:
            objectives.append(current_obj)
            segments.append('\n'.join(current_seg))
            keywords_list.append(current_keys)

        return {
            "num_objectives": n,
            "objectives": objectives,
            "segments": segments,
            "keywords": keywords_list
        }


###############################################################
# Agent2:Determine the question type based on the text and generate the draft question
###############################################################
# Question type criteria:
# Conceptual Questions: Used to understand concepts, principles, and meanings.
# Numerical Questions: involves numerical operations such as numbers, ASCII values, percentages, and position calculations.
# Application Questions: Apply the knowledge learned to specific operations, such as decryption, plaintext finding from a given ciphertext, verification process, etc.

class QuestionGenerationAgent:
    def __init__(self, name="QuestionGenerationAgent"):
        self.name = name

    def generate_question(self, segment_text, objective, keywords):
        prompt = (
            "You are an expert educational AI assistant.\n"
            "Your tasks:\n"
            "1. Determine which of the following three types of questions would be most beneficial for achieving the teaching objective, based on the provided text segment, objective, and keywords.\n"
            "2. After determining the question type, generate an initial draft question that matches the chosen type and helps achieve the learning objective.\n\n"

            "Question Types and Their Standards:\n"
            "- Conceptual Questions: Focus on understanding concepts, principles, or the reasoning behind processes. Use this when the objective and text emphasize understanding 'what', 'why', or 'how' something works conceptually.\n"
            "example of Conceptual Questions: What is the range of keys in a Caesar Cipher?\n"
            "- Numerical Questions: Involve direct numeric computations, ASCII values, letter positions, frequencies, percentages, or key values. Use this when the text and keywords strongly hint at numeric relationships or calculations.\n"
            "example of Numerical Questions: What is the ASCII value of the letter 'Y'?\n"
            "- Application Questions: Involve applying the learned concept to a concrete scenario, such as decrypting a given ciphertext, demonstrating brute-forcing, or using frequency analysis to determine a plaintext letter. Use this if the objective suggests practical usage or scenario-based tasks. If there are mathematical formulas in the text, priority should be given to application question\n"
            "The generated questions should demonstrate specific application scenarios, example of Application Questions: Given the ciphertext 'AWW' and a key of 2, apply the Caesar Cipher to find the original plain.\n\n"

            "Rules:\n"
            "1. All questions are Q&A type, no multiple-choice, no fill-in.\n\n"

            "attention:\n"
            "1. Output strictly according to format, specially attention to the use of newline characters\n"
            "2. Do not output explanations beyond the required format as much as possible.\n\n"

            "Output Format:\n"
            "Question Type: [Question Type]\n<END>\n"
            "Draft Question: [Draft Question]\n<END>\n\n"

            "Example Input:\n"
            "Objective: Understand the concept of Caesar Cipher\n"
            "Keywords: cipher, shift, alphabet\n"
            "Segment: \"This segment explains the Caesar Cipher and why shifting letters by a given key changes their position in the alphabet.\"\n"
            "Example Output:\n"
            "Question Type: Conceptual Questions\n"
            "<END>\n"
            "Draft Question: Explain how the Caesar Cipher uses modular arithmetic (mod 26) to shift letters.\n"
            "<END>\n\n"
            
            "Now, based on these standards, determine the best question type and output it.\n"
            "Then generate a draft question of that type aligned with the objective and the segment.\n"
            "Remember to follow the Output Format strictly.\n"
            f"Objective: {objective}\n"
            f"Keywords: {', '.join(keywords)}\n"
            f"Segment:\n\"{segment_text}\"\n\n"
        )

        response = pipe(prompt, max_new_tokens=300, return_full_text=False)[0]['generated_text']
        parts = response.split("<END>")
        parts = [p.strip() for p in parts if p.strip()]
        chosen_type = "Conceptual Questions"  
        draft_question = ""
        found_type = False
        found_draft = False
        for p in parts:
            lines = p.split('\n')
            for line in lines:
                line_stripped = line.strip().lower()
                if line_stripped.startswith("question type:"):
                    line_parts = line.split(":", 1)
                    if len(line_parts) > 1:
                        t = line_parts[1].strip()
                        if t in ["Conceptual Questions", "Numerical Questions", "Application Questions"]:
                            chosen_type = t
                            found_type = True
                elif line_stripped.startswith("draft question") and found_type:
                    line_parts = line.split(":", 1)
                    if len(line_parts) > 1:
                        d = line_parts[1].strip()
                        draft_question = d
                        found_draft = True
                        break
            if found_type and found_draft:
                break


        if not found_type:
            chosen_type = "Conceptual Questions"
        if not draft_question:
            draft_question = "Explain the core concept behind the given cipher."
        return chosen_type, draft_question



###############################################################
# Agent3：Evaluate the question and give suggestions
###############################################################
class QuestionEvaluationAgent:
    def __init__(self, name="QuestionEvaluationAgent"):
        self.name = name

    def evaluate_question(self, segment_text, objective, question, keywords):
        prompt = (
            "You are an AI teaching expert.\n"
            "Evaluate the draft question in terms of clarity, difficulty, and relevance, then provide one suggestion.\n\n"
            "Output format:\n"
            "Feedback: [one sentence]\n"
            "Suggestion: [one sentence]\n"
            "<END>\n\n"
            "Example:\n"
            "Draft Question: \"Explain how Caesar Cipher works.\"\n"
            "Feedback: The question is clear and relevant.\n"
            "Suggestion: Add more specificity about the shifting process.\n"
            "<END>\n\n"
            f"Objective: {objective}\n"
            f"Keywords: {', '.join(keywords)}\n"
            f"Draft Question:\n\"{question}\"\n\n"
            "Segment:\n\"{segment_text}\"\n\n"
            "Now output feedback and suggestion followed by <END>."
        )
        response = pipe(prompt, max_new_tokens=200, return_full_text=False)[0]['generated_text']
        if "<END>" in response:
            response = response.split("<END>")[0].strip()
        lines = response.split('\n')
        feedback = ""
        suggestion = ""
        for line in lines:
            if line.lower().startswith("feedback:"):
                feedback = line.split(':',1)[1].strip()
            elif line.lower().startswith("suggestion:"):
                suggestion = line.split(':',1)[1].strip()
        return feedback, suggestion

###############################################################
# QuestionRefinementAgent：Modify the question as suggested
###############################################################
class QuestionRefinementAgent:
    def __init__(self, name="QuestionRefinementAgent"):
        self.name = name

    def refine_question(self, draft_question, suggestion, segment_text, objective, keywords):
        prompt = (
            "You are an AI assistant that refines the draft question based on the suggestion of the evaluation agent.\n\n"

            "attention:\n"
            "1. Output strictly according to format, specially attention to the use of newline characters\n"
            "2. Do not output explanations beyond the required format.\n\n"

            "Output format:\nRevised Question:\n[Revised Question]\n<END>\n\n"

            "Example:\n"
            "Draft Question: \"Explain Caesar Cipher.\"\n"
            "Suggestion: Specify how the key affects letter shifting.\n"
            "Revised Question: How does the chosen key affect the letter shifting process in a Caesar Cipher?\n"
            "<END>\n\n"

            "Please produce the revised question and then <END>."
            f"Draft Question:\n\"{draft_question}\"\n"
            f"Suggestion:\n\"{suggestion}\"\n"
            f"Segment:\n\"{segment_text}\"\n\n"
        )
        response = pipe(prompt, max_new_tokens=200, return_full_text=False)[0]['generated_text']
        if "<END>" in response:
            response = response.split("<END>")[0].strip()
        else:
            response = response.strip()
        lines = response.split('\n')
        revised_question = ""
        found_revised_line = False

        for i, line in enumerate(lines):
            if not found_revised_line and line.lower().startswith("revised question:"):
                found_revised_line = True
                line = line.split(':',1)[1].strip()
            if found_revised_line:
                revised_question += line.strip()
        return revised_question

###############################################################
# AnswerGenerationAgent：，Generate answers based on text and questions
###############################################################
class AnswerGenerationAgent:
    def __init__(self, name="AnswerGenerationAgent"):
        self.name = name

    def generate_answer(self, segment_text, objective, question, keywords):
        prompt = (
            "You are an AI assistant generating an answer based only on the provided segment.\n"
            "Output format:\nAnswer: [Answer]\n<END>\n\n"

            "attention:\n"
            "1. Output strictly according to format, specially attention to the use of newline characters\n"
            "2. Do not output explanations beyond the required format.\n\n"

            "Example:\n"
            "Question: \"How does Caesar Cipher shift letters?\"\n"
            "Segment: \"...explains shifting letters by key...\"\n"
            "Answer:\n\"The Caesar Cipher shifts letters by adding the key to their alphabetical index, wrapping around if needed.\"\n"
            "<END>\n\n"
            
            "Now produce the answer and then <END>."
            f"Question:\n\"{question}\"\n"
            f"Segment:\n\"{segment_text}\"\n\n"
        )

        response = pipe(prompt, max_new_tokens=300, return_full_text=False)[0]['generated_text']
        if "<END>" in response:
            response = response.split("<END>")[0].strip()
        else:
            response = response.strip()

        lines = response.split('\n')
        answer = ""
        found_answer_line = False

        for i, line in enumerate(lines):
            if not found_answer_line and line.lower().startswith("answer:"):
                found_answer_line = True
                line = line.split(':',1)[1].strip()
            if found_answer_line:
                answer += line.strip()
        return answer

###############################################################
# Main process
###############################################################
def main():
    for i, large_text in enumerate(large_texts):
        print(f"\nProcessing large_text {i+1}/{len(large_texts)}...")

	# Analyze the text to get targets, text segments, and keywords
        analysis_agent = MaterialAnalysisAndSegmentationAgent()
        analysis_result = analysis_agent.analyze_and_segment(large_text)
        n = analysis_result["num_objectives"]
        objectives = analysis_result["objectives"]
        segments = analysis_result["segments"]
        keywords_list = analysis_result["keywords"]
    
        print(f"Number of Objectives: {n}")
    
        # 4 agents generate questions, evaluate questions, modify questions, and generate answers
        question_generation_agent = QuestionGenerationAgent()  
        q_eval_agent = QuestionEvaluationAgent()
        q_refine_agent = QuestionRefinementAgent()
        ans_agent = AnswerGenerationAgent()
    
        # Loop generates questions for each learning goal
        for j in range(n):
            objective = objectives[j]
            segment_text = segments[j]
            kws = keywords_list[j]
    
            # Generate the type of question and generate the preliminary question
            q_type, draft_q = question_generation_agent.generate_question(segment_text, objective, kws)
            
            # Evaluate and improve the problem
            feedback, suggestion = q_eval_agent.evaluate_question(segment_text, objective, draft_q, kws)
            refined_q = q_refine_agent.refine_question(draft_q, suggestion, segment_text, objective, kws)
            
            # Generate answers
            answer = ans_agent.generate_answer(segment_text, objective, refined_q, kws)

            # Print details for each chapter
            chapter_index = i + 1
            subchapter_index = j + 1
            print(f"\n=== Chapter {chapter_index} - Subchapter {subchapter_index} ===")
            print("Objective:", objective)
            print("Segment Text:", segment_text)
            print("Keywords:", kws)
            print("Question Type:", q_type)
            print("Question:", refined_q)
            print("Answer:", answer)

if __name__ == "__main__":
    main()