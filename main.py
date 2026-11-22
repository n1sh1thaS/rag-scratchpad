from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

def main():
    print("Hello from rag-scratchpad!")
    llm_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    response = llm_openai.invoke("Say 'setup complete' in one word")
    print(f"Response from ChatOpenAI: {response}")

    llm_anthropic = ChatAnthropic(model_name="claude-sonnet-4-5-20250929", temperature=0)
    response = llm_anthropic.invoke("Say 'setup complete' in one word")
    print(f"Response from ChatAnthropic: {response}")

if __name__ == "__main__":
    main()
