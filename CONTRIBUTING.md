# Contributing to EiN

Bug reports, ideas and code are welcome. This file says where to send them, what the code
should look like, and what you agree to when you contribute.

## Where to send it

Open issues and pull requests in this repository on GitHub, in the
[einhq](https://github.com/einhq) organisation. For anything larger than a small fix, open an
issue first, so that the approach is agreed before you spend time on it.

A pull request is squash-merged: it becomes one commit on `main`, whose message is the pull
request's title and description. The title therefore follows the commit message rules below,
and a check on the pull request says whether it does.

## The licence and the contributor licence agreement

This repository is under the Elastic License 2.0 with an added permission, as
[LICENSE](LICENSE) says. Its author also sells separate licences for hosting EiN as a service,
so before your first contribution is accepted you agree to the
[EiN Contributor License Agreement](CLA.md). You keep the copyright to what you contribute;
you give the licensor the right to use it and to license it on any terms, including those
separate licences.

When you open your first pull request, the CLA assistant comments on it with a link; sign the
agreement there with your GitHub account. A pull request is merged only once everyone with a
commit in it has signed. Signing once covers every later contribution you make to any of EiN's
repositories under the Elastic License. If what you create belongs to your employer, such as
code you write at work, your employer has to allow it first (section 5 of the agreement).

The libraries EiN is built on, bash-parser, bash-executor and ssh2, are under the MIT licence
and need no agreement.

## How the code is written

- **Commit messages**, and so pull request titles, follow
  [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `fix(ems): …`,
  `feat: …`, `docs: …`, with `!` and a `BREAKING CHANGE:` footer for a breaking change. The description is one line, without a trailing period, saying what the commit does.
- **Comments** describe the code as it is now: what it does, takes, returns and guarantees.
  Leave out its history, what it replaced and what was tried first; that belongs in the commit
  message.
- **Unused code** is deleted, not commented out or kept "in case": git keeps old versions. A
  change that removes a feature also removes its tests, scripts and documentation.
- **Documentation**: a change a user could notice, such as a command, an option, a setting or
  a page, updates the documentation at [einhq.eu/docs](https://einhq.eu/docs/). It lives in
  [einhq.github.io](https://github.com/einhq/einhq.github.io); send a pull request there too.
  Examples use invented people, never real ones.
- **Tests**: add or change tests with the code. The README says how to run, test and check this
  repository; a module is developed and tested inside a checkout of EiN core.

## Security problems

Do not report a security problem in a public issue. Email it to Mattias Runge-Broberg,
<mattias@runge.se>, instead.
