# Contributing to `ennemi`

Thank you for your interest in contributing to `ennemi` development, awesome!
Please read through the following quick points on the workflow.


## Code of Conduct
Everybody is welcome to contribute to this package
regardless of their personal or academic background.
Criticism is encouraged as long as it is constructive and respectful.
Everybody makes mistakes; I think safely erring is a great way to learn.

This project is maintained by **Petri Laarne**, `firstname.lastname @ helsinki.fi`.
Feel free to contact me on anything related to using or contributing to this project.

As the project maintainer, I pledge to follow
[Contributor Covenant 2.0](https://www.contributor-covenant.org/version/2/0/code_of_conduct/)
and expect other contributors to do the same.

> Examples of behavior that contributes to a positive environment for our community include:
> 
> - Demonstrating empathy and kindness toward other people
> - Being respectful of differing opinions, viewpoints, and experiences
> - Giving and gracefully accepting constructive feedback
> - Accepting responsibility and apologizing to those affected by our mistakes, and learning from the experience
> - Focusing on what is best not just for us as individuals, but for the overall community

As of this writing, the citation for `ennemi` includes Petri Laarne as the author.
If you contribute to the package, the citation will be changed to
"Petri Laarne and contributors" for future releases.
This policy may be updated.


## Submitting issues
All feedback is much appreciated, no matter how big or small.
Especially in the current alpha/beta phase it is easy to make the package more ergonomic to use.
Comments on documentation and small pain points ("papercut issues") are really useful there.

To make problems easier to investigate:

- Clearly tell what you are trying to do, what you expect `ennemi` to do and what `ennemi` actually does.
- If you can attach example code, please do so.
  Try to make the example minimal and self-contained: remove all lines that are not
  relevant for the problem, and make any datasets as small as possible
  (or replace them with constants or random numbers).


## Submitting Pull Requests
Code changes are welcomed as well.
Before submitting larger changes, please comment on or create an issue
for discussion on the direction to take.
You don't need to create an issue for a simple bug fix,
but it is okay as well.
Do mention that you are interested in submitting a fix,
and feel free to ask for help.

Pull requests will go through a code review.
You may need to address review comments before the change is merged.
These comments are open for discussion and disagreement.
The goal is to make the package as good as possible.
At the same time, "perfect is the enemy of good"; excessive bikeshedding
and polishing is discouraged.

There are a few technical details you should be aware of.

### Code and commit style
At the moment, the code is not linted.
Try to keep the lines shorter than 100 characters,
preferably less than 80.
Try to preserve the existing style of files.
General Python style should be followed.

For Markdown files, the same line length should be followed.
Prefer starting each sentence on its own line.
This makes Git diffs easier to read.

Commit messages must not exceed 50 characters, must start with a capital letter
and must be in present imperative mood.
For example:

- GOOD: "Update release script to point to live PyPI"
- BAD: "updated release script to point to live PyPI" (past tense, small letter)
- BAD: "Release script now points to live PyPI" (not imperative mood)

Commit descriptions are not currently used, but are certainly allowed
to give more information.
For more information on good Git commit messages, see the excellent article
[How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/).


### Unit tests
The package has an extensive unit test suite guarding against regressions.
Test-driven development is strongly encouraged.
This means that the process of fixing a bug or creating a new feature is:

1. Write a unit test for the bug / a piece of feature.
2. Run the test and see that it fails.
3. Fix the bug / implement the piece of feature.
4. Run the test and see that it passes.
5. Repeat until done.

Each pull request is tested by a Continuous Integration bot.
The automated test run also includes static analysis of code quality.
Unit test coverage is measured by the script,
and too low coverage fails the build.

The goal is that the tests fully document the package.
When reviewing a Pull Request, the code changes should be obvious from the new/modified tests.
If the tests haven't changed, the assumption is that the behavior hasn't changed either.


### Type checking
The package uses [PEP 484](https://www.python.org/dev/peps/pep-0484/)
type hints in all code, checked by `mypy`.
This also includes the test code for completeness.
Failing type check fails the automated build.

You should run the type check manually before pushing changes,
unless you really trust your type annotations.
See the README file for details.



## The Zen of `ennemi`
- Practicality is more important than theoretical completeness.
- The 95% use case is more important than the special cases.
- Although special uses should be achievable too.
- Performance is important.
- But not as important as correctness, usability and portability.
