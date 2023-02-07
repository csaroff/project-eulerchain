from pathlib import Path

# ## Scraping the speech from projecteuler.net
#
# It's super easy to scrape the problem text from project euler. Each problem has its own page.
# We just have to ensure that we're preserving links and special syntax like subscript and superscript.
def scrape_euler_problem(problem_number: int) -> str:
    import httpx
    from bs4 import BeautifulSoup

    url = f"https://projecteuler.net/problem={problem_number}"

    # fetch article; simulate desktop browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"
    }
    response = httpx.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "lxml")

    # get all text paragraphs & construct string of article text
    title = soup.find_all("h2")
    markdown = title[0].get_text() + ": "

    paragraphs = soup.find_all('p')[:-1]

    for p in paragraphs:
        # Loop through each sub_element
        for sub_element in p.children:

            # Check if the sub_element is a link
            if sub_element.name == 'a':
                # Add the link to the markdown
                markdown += f"[{sub_element.get_text()}]({sub_element['href']})"
            # If the sub_element is text, preserve it as is
            elif sub_element.name is None:
                markdown += sub_element
            # If the sub_element is subscript, preserve it as is
            elif sub_element.name == "sub":
                markdown += f"<sub>{sub_element.text}</sub>"
            # If the sub_element is superscript, preserve it as is
            elif sub_element.name == "sup":
                markdown += f"<sup>{sub_element.text}</sup>"
            # Check if the sub_element is bold text
            elif sub_element.name == 'strong':
                # Add the bold text to the markdown
                markdown += f"**{sub_element.get_text()}**"
            # Check if the sub_element is italic text
            elif sub_element.name == 'em':
                # Add the italic text to the markdown
                markdown += f"*{sub_element.get_text()}*"
            # Check if the sub_element is code text
            elif sub_element.name == 'code':
                # Add the code text to the markdown
                markdown += f"`{sub_element.get_text()}`"
        markdown += " "
    return markdown.replace("\t", "")


def solve_euler_problem(problem_number: int) -> tuple[str, list[str]]:
    from langchain.llms import OpenAI
    from langchain.chains import PALChain
    from langchain.text_splitter import CharacterTextSplitter

    # Support caching speech text on disk.
    euler_problems_path = Path("./euler_problems/")
    euler_problems_path.mkdir(exist_ok=True)
    print("problem number", problem_number)
    print(f"{str(problem_number).zfill(4)}")
    euler_problem_path = euler_problems_path/f"{str(problem_number).zfill(4)}.txt"

    if euler_problem_path.exists():
        euler_problem = euler_problem_path.read_text()
    else:
        print(f"Scraping Project Euler problem {problem_number}")
        euler_problem = scrape_euler_problem(problem_number)
        euler_problem_path.write_text(euler_problem)

    # We cannot send the entire problem to the model because OpenAI's model
    # has a maximum limit on input tokens. So we split up the speech
    # into smaller chunks.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    print("splitting problem into text chunks")
    texts = text_splitter.split_text(euler_problem)

    llm = OpenAI(model_name='code-davinci-002', temperature=0, max_tokens=512)
    pal_chain = PALChain.from_math_prompt(llm, verbose=True)
    print("running query against programming aided llm chain")

    return pal_chain.run(euler_problem)


print("Solving euler problem:")
print(solve_euler_problem(1))
