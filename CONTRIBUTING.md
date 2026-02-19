# Contributing to 0xBitNet

Thank you for your interest in contributing to 0xBitNet! This document provides guidelines to help you get started.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- A clear title and description
- Steps to reproduce the issue
- Expected vs actual behavior
- Browser and GPU information (if relevant)

### Suggesting Features

Feature requests are welcome. Please open an issue describing:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
   ```bash
   git checkout -b feature/your-feature
   ```
3. Make your changes
4. Ensure your code passes linting and tests
   ```bash
   npm run lint
   npm run test
   ```
5. Commit with a clear message
   ```bash
   git commit -m "Add: brief description of your change"
   ```
6. Push and open a Pull Request against `main`

## Development Setup

```bash
git clone https://github.com/m96-chan/0xBitNet.git
cd 0xBitNet
npm install
npm run dev
```

## Code Style

- TypeScript for all source code
- WGSL for GPU kernels
- Use the existing code style — the linter will catch most issues

## Commit Message Convention

Use a short prefix to categorize your commit:

- `Add:` — New feature or file
- `Fix:` — Bug fix
- `Update:` — Enhancement to existing functionality
- `Refactor:` — Code restructuring without behavior change
- `Docs:` — Documentation only
- `Test:` — Adding or updating tests

## Areas Where Help Is Needed

- WGSL kernel optimization and correctness
- Browser compatibility testing
- Documentation and examples
- Performance benchmarking

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
