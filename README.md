# twiga

Twiga is a Whatsapp chatbot for Tanzanian teachers. I am creating this as part of my master thesis project at KTH so some of the content is specific to testing and implementing a research-based RAG pipeline and some is specific to deployment on Whatsapp.

## Commit message convention

When writing commit messages for this project, please try to write your commit messages in the following manner:

_Keyword(scope if needed): write the message in the imperative sense without capitals and no full stop at the end_

### Possible keywords

- Build: Build related changes (eg: npm related/ adding external dependencies)
- Chore: A code change that external user won't see (eg: change to .gitignore file or .prettierrc file)
- Feat: A new feature
- Fix: A bug fix
- Docs: Documentation related changes
- Refactor: A code that neither fix bug nor adds a feature. (eg: You can use this when there is semantic changes like renaming a variable/ function name)
- Perf: A code that improves performance
- Style: A code that is related to styling
- Test: Adding new test or making changes to existing test
- Misc: Anything that in no way fits into the other categories

## Using a virtual environment

When using this repo, it is recommended to use a virtual environment and to import the packages from `requirements.txt`. This is done simply by importing the relevant dependencies using `pip install -r requirements.txt`. If you want to save the current dependencies list you have in your .venv in `requirements.txt` you can just write `pip freeze > requirements.txt`.
