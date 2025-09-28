# -*- coding: utf-8 -*-
"""
Reward functions with LLM as a judge

This module demonstrates how to use LLMs as judges for evaluating text summaries
using multiple reward mechanisms including direct scoring, quiz-based evaluation,
and length penalties.
"""

import os
import re
from random import shuffle
from typing import List, Tuple, Optional

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from predibase import Predibase
from tabulate import tabulate
from matplotlib import pyplot as plt

# Set API keys
load_dotenv()

# =====================================
# CONFIGURATION AND CLIENT SETUP
# =====================================

def setup_environment() -> Tuple[OpenAI, OpenAI]:
    """
    Set up environment variables and initialize API clients.
    
    Returns:
        Tuple of (pb_client, openai_client) for Predibase and OpenAI respectively
    """
    # Predibase client setup
    tenant_id = "b2793b72"
    deployment_name = "llama-3-1-8b-instruct"
    api_token = os.environ["PREDIBASE_API_KEY"]
    base_url = f"https://serving.app.predibase.com/{tenant_id}/deployments/v2/llms/{deployment_name}/v1"
    
    pb_client = OpenAI(api_key=api_token, base_url=base_url)
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    return pb_client, openai_client


# =====================================
# DATA MODELS
# =====================================

class Question(BaseModel):
    """Represents a single multiple-choice question in a quiz."""
    text: str
    options: List[str]
    answer: int

    def shuffle_options(self) -> None:
        """Shuffle the options while preserving the correct answer."""
        correct = self.options[self.answer]
        shuffled = self.options.copy()
        shuffle(shuffled)
        self.options = shuffled
        self.answer = shuffled.index(correct)

    def __str__(self) -> str:
        """Format question for display."""
        output = [self.text]
        for i, option in enumerate(self.options):
            output.append(f"{chr(65+i)}. {option}")
        return "\n".join(output)


class Quiz(BaseModel):
    """Represents a complete quiz with multiple questions."""
    questions: List[Question]

    def shuffle_all_questions(self) -> None:
        """Shuffle the options for all questions in the quiz."""
        for question in self.questions:
            question.shuffle_options()

    def __str__(self) -> str:
        """Format entire quiz for display."""
        output = []
        for i, question in enumerate(self.questions, 1):
            output.append(f"\nQuestion {i}:")
            output.append(str(question))
        return "\n".join(output)


# =====================================
# UTILITY FUNCTIONS
# =====================================

def compute_advantages(rewards: List[float]) -> List[float]:
    """
    Compute advantage scores by normalizing rewards.
    
    Args:
        rewards: List of reward scores
        
    Returns:
        List of advantage scores (standardized rewards)
    """
    rewards_array = np.array(rewards)
    mean_reward = np.mean(rewards_array)
    std_reward = np.std(rewards_array)
    
    # Avoid division by zero
    if std_reward == 0:
        return [0.0] * len(rewards)
    
    advantages = (rewards_array - mean_reward) / std_reward
    return advantages.tolist()


# =====================================
# DISPLAY FUNCTIONS
# =====================================

def print_rewards_table(rewards: List[float], title: str = "Rewards") -> None:
    """Print a formatted table of rewards and advantages."""
    advantages = compute_advantages(rewards)
    elems = list(zip(range(len(rewards)), rewards, advantages))
    headers = ["Index", "Reward", "Advantage"]
    table = tabulate(elems, headers=headers, tablefmt="grid")
    print(f"\n{title}")
    print(table)


def print_length_table(lengths: List[int], rewards: List[float]) -> None:
    """Print a table showing lengths, rewards, and advantages."""
    advantages = compute_advantages(rewards)
    elems = list(zip(range(len(lengths)), lengths, rewards, advantages))
    headers = ["Index", "Length", "Reward", "Advantage"]
    table = tabulate(elems, headers=headers, tablefmt="grid")
    print("\nLength Analysis")
    print(table)


def print_total_rewards_table(
    length_rewards: List[float], 
    quiz_rewards: List[float], 
    total_rewards: List[float]
) -> None:
    """Print comprehensive table showing all reward components."""
    advantages = compute_advantages(total_rewards)
    elems = list(zip(
        range(len(length_rewards)),
        length_rewards,
        quiz_rewards,
        total_rewards,
        advantages
    ))
    headers = ["Index", "Length Reward", "Quiz Reward", "Total Reward", "Advantage"]
    table = tabulate(elems, headers=headers, tablefmt="grid")
    print("\nTotal Rewards Analysis")
    print(table)


def visualize_reward_tradeoff(
    length_rewards: List[float], 
    quiz_rewards: List[float], 
    advantages: List[float]
) -> None:
    """Create a scatter plot showing the trade-off between length and quiz rewards."""
    min_adv, max_adv = min(advantages), max(advantages)
    
    plt.figure(figsize=(10, 6), facecolor='black')
    plt.style.use('dark_background')
    
    scatter = plt.scatter(
        length_rewards, quiz_rewards, 
        c=advantages, cmap='RdYlGn', s=100, 
        edgecolor='white', vmin=min_adv, vmax=max_adv
    )
    
    plt.colorbar(scatter, label='Advantage')
    plt.xlabel('Length Reward')
    plt.ylabel('Quiz Reward')
    plt.title('Length Reward vs Quiz Reward (colored by advantage)')
    plt.grid(True, alpha=0.2)
    plt.show()


# =====================================
# CORE FUNCTIONALITY CLASSES
# =====================================

class SummaryGenerator:
    """Handles text summarization using LLM."""
    
    SUMMARIZE_PROMPT = """Generate a concise summary of the information in the following earnings call transcript.

Only respond with the summary, do not include any extraneous text.

Transcript:

{transcript}
"""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def generate_summary(self, transcript: str, n: int = 1, temperature: float = 0.9) -> List[str]:
        """
        Generate one or more summaries of the given transcript.
        
        Args:
            transcript: Input text to summarize
            n: Number of summaries to generate
            temperature: Sampling temperature for generation
            
        Returns:
            List of generated summaries
        """
        prompt = self.SUMMARIZE_PROMPT.format(transcript=transcript)
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat.completions.create(
            model="",  # leave blank for deployment
            messages=messages,
            n=n,
            temperature=temperature,
            max_tokens=256,
        )
        
        return [choice.message.content for choice in response.choices]


class RewardJudge:
    """Handles LLM-based evaluation of summaries."""
    
    JUDGE_PROMPT = """Rate the following summary of an earnings call transcript on a scale from 1 to 10.

1 means the summary is very poor, 10 means the summary is very good.

Provide reasoning followed by the final score at the end surrounded by <score> tags.

For example: <score>1</score>

Transcript:
{transcript}

Summary:
{summary}
"""
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
    
    def judge_summary(self, transcript: str, summary: str, verbose: bool = False) -> float:
        """
        Evaluate a summary using LLM as judge.
        
        Args:
            transcript: Original transcript
            summary: Summary to evaluate
            verbose: Whether to print judge reasoning
            
        Returns:
            Normalized score (0.0 to 1.0)
        """
        prompt = self.JUDGE_PROMPT.format(transcript=transcript, summary=summary)
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            n=1,
            temperature=0,
        )
        
        completion = response.choices[0].message.content
        
        if verbose:
            print(completion)
        
        try:
            match = re.search(r"<score>(\d+)</score>", completion)
            if match is None:
                return 0.0
            score = int(match.group(1).strip())
        except (ValueError, AttributeError):
            score = 0
        
        return score / 10.0


class QuizGenerator:
    """Handles quiz creation and evaluation."""
    
    QUIZ_PROMPT = """Generate a multiple-choice quiz based on the information in the following earnings call transcript.

Example:
```
1. What was the q1 adjusted earnings per share?
a) $3.34
b) $5.32
c) $2.49
d) $7.78

2. By what percent did same store sales rise in q1?
a) 29.4%
b) 32.1%
c) 24.7%
d) 21.2%

===== ANSWERS =====
1. a
2. c
```

Limit the length of the quiz to the top 10 most relevant questions for financial analysts.

Transcript:
{transcript}
"""
    
    TAKE_QUIZ_PROMPT = """Use the provided summary of a transcript to answer the following quiz.

Quiz:
{quiz}

Summary:
{summary}

Respond with just a list of answers and no additional text, for example:
[A, D, C, B, B, C, D, A, A, B]

You must provide an answer for all questions.
If you don't know the answer, answer with "0" for that question.
Example: [A, D, 0, B, B, C, D, A, A, B]
"""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.index_to_letter = ["A", "B", "C", "D"]
    
    def create_quiz(self, transcript: str) -> Quiz:
        """
        Create a quiz based on transcript content.
        
        Args:
            transcript: Source material for quiz questions
            
        Returns:
            Quiz object with validated questions
        """
        prompt = self.QUIZ_PROMPT.format(transcript=transcript)
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            response_format=Quiz,
        )
        
        quiz = response.choices[0].message.parsed
        quiz.shuffle_all_questions()
        
        # Validate questions by testing on original transcript
        quiz = self._validate_quiz_questions(transcript, quiz)
        
        # Limit to 10 questions
        quiz.questions = quiz.questions[:10]
        
        return quiz
    
    def _validate_quiz_questions(self, transcript: str, quiz: Quiz) -> Quiz:
        """Remove questions that can't be answered correctly using the transcript."""
        prev_len = len(quiz.questions)
        
        while True:
            answers = self.take_quiz(transcript, quiz)
            answerable_questions = []
            
            for answer, question in zip(answers, quiz.questions):
                expected_answer = self.index_to_letter[question.answer]
                if answer == expected_answer:
                    answerable_questions.append(question)
            
            quiz.questions = answerable_questions
            
            if len(quiz.questions) == prev_len:
                break
            prev_len = len(quiz.questions)
        
        return quiz
    
    def take_quiz(self, summary: str, quiz: Quiz) -> List[str]:
        """
        Have LLM take quiz using provided summary.
        
        Args:
            summary: Text to use for answering questions
            quiz: Quiz to take
            
        Returns:
            List of answer letters
        """
        quiz_str = self._format_quiz_for_taking(quiz)
        prompt = self.TAKE_QUIZ_PROMPT.format(quiz=quiz_str, summary=summary)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        resp_str = response.choices[0].message.content
        answers = resp_str.strip('[]').split(', ')
        
        return answers
    
    def _format_quiz_for_taking(self, quiz: Quiz) -> str:
        """Format quiz questions for the take_quiz prompt."""
        question_strs = []
        for question in quiz.questions:
            question_str = question.text
            for i, option in enumerate(question.options):
                letter = self.index_to_letter[i]
                question_str += f"\n{letter}. {option}"
            question_strs.append(question_str)
        return "\n\n".join(question_strs)
    
    def score_quiz(self, answers: List[str], quiz: Quiz) -> float:
        """
        Score quiz answers against correct answers.
        
        Args:
            answers: List of answer letters from quiz taker
            quiz: Quiz with correct answers
            
        Returns:
            Score as fraction correct (0.0 to 1.0)
        """
        if len(answers) != len(quiz.questions):
            return 0.0
        
        correct = 0
        for answer, question in zip(answers, quiz.questions):
            expected_answer = self.index_to_letter[question.answer]
            if answer == expected_answer:
                correct += 1
        
        return correct / len(answers)


class RewardCalculator:
    """Handles different types of reward calculations."""
    
    @staticmethod
    def length_penalty_reward(response: str, target_length: int = 1024) -> float:
        """
        Calculate length penalty for responses exceeding target length.
        
        Args:
            response: Text to evaluate
            target_length: Target character count
            
        Returns:
            Penalty score (0.0 for acceptable length, negative for excess)
        """
        length = len(response)
        if length <= target_length:
            return 0.0
        else:
            return max((target_length - length) / target_length, -10.0)
    
    @staticmethod
    def total_reward(length_reward: float, quiz_reward: float) -> float:
        """Combine length and quiz rewards."""
        return length_reward + quiz_reward


# =====================================
# MAIN EXECUTION
# =====================================

def main():
    """Main execution function demonstrating the reward system."""
    
    # Setup
    pb_client, openai_client = setup_environment()
    
    # Load data
    ds = load_dataset("mrSoul7766/ECTSum")
    transcript = ds["train"][1]["text"]
    print("Transcript preview:")
    print(transcript[:1983])
    
    # Initialize components
    summarizer = SummaryGenerator(pb_client)
    judge = RewardJudge(openai_client)
    quiz_gen = QuizGenerator(openai_client)
    
    # Generate summaries
    print("\n" + "="*50)
    print("GENERATING SUMMARIES")
    print("="*50)
    
    summaries = summarizer.generate_summary(transcript, n=8)
    
    # Test direct LLM judging
    print("\n" + "="*50)
    print("DIRECT LLM JUDGING")
    print("="*50)
    
    judge_scores = [judge.judge_summary(transcript, summary) for summary in summaries]
    print_rewards_table(judge_scores, "Direct Judge Scores")
    
    # Create and test quiz
    print("\n" + "="*50)
    print("QUIZ-BASED EVALUATION")
    print("="*50)
    
    quiz = quiz_gen.create_quiz(transcript)
    print("Generated Quiz:")
    print(quiz)
    
    quiz_scores = []
    for summary in summaries:
        answers = quiz_gen.take_quiz(summary, quiz)
        score = quiz_gen.score_quiz(answers, quiz)
        quiz_scores.append(score)
    
    print_rewards_table(quiz_scores, "Quiz Scores")
    
    # Test length penalties
    print("\n" + "="*50)
    print("LENGTH PENALTY ANALYSIS")
    print("="*50)
    
    lengths = [len(summary) for summary in summaries]
    length_rewards = [RewardCalculator.length_penalty_reward(summary) for summary in summaries]
    
    print_length_table(lengths, length_rewards)
    
    # Combined rewards
    print("\n" + "="*50)
    print("COMBINED REWARD ANALYSIS")
    print("="*50)
    
    total_rewards = [
        RewardCalculator.total_reward(length_reward, quiz_reward)
        for length_reward, quiz_reward in zip(length_rewards, quiz_scores)
    ]
    
    print_total_rewards_table(length_rewards, quiz_scores, total_rewards)
    
    # Visualization
    advantages = compute_advantages(total_rewards)
    visualize_reward_tradeoff(length_rewards, quiz_scores, advantages)


if __name__ == "__main__":
    main()
