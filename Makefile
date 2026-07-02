# Weightless — dev & release tasks
#
# Releases are tag-triggered: `make release` bumps the version (which creates
# the vX.Y.Z tag) and pushes it; CI (.github/workflows/release.yml) then tests,
# packages, creates the GitHub Release, and publishes to the VS Code
# Marketplace + Open VSX. Ordinary pushes never release anything.

.PHONY: test package release release-minor release-major

test:
	npm test

package: test
	npx --yes @vscode/vsce package

release: test              ## bug fixes: 0.1.2 -> 0.1.3
	npm version patch
	git push
	git push --tags

release-minor: test        ## new features: 0.1.2 -> 0.2.0
	npm version minor
	git push
	git push --tags

release-major: test        ## breaking changes: 0.1.2 -> 1.0.0
	npm version major
	git push
	git push --tags
