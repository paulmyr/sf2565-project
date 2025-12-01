# sf2565-project

## Setup Code
### 1. Run git submodule update to install Eigen library

```bash
git submodule update --init --recursive
```

### 2. Build the project

```bash
mkdir -p build
cd build
cmake ..
make
```

### 3. Run using
```bash
./sf2565_project --help
./sf2565_project --input-file <path-to-file>
```

## Setup CLion
1. Clone the project to your machine (with submodules as shown above)
2. Start CLion → New Project → Navigate to folder → Open → Project from existing sources
3. CLion should automatically detect CMakeLists.txt and configure the project
4. If you encounter indexing issues, go to: Fiel → Invalidate Caches → Invalidate and REstart

## Dependencies

- **Eigen** (linear algebra library) - in `lib/eigen/`
- **CLI11** (command-line parsing) - in `lib/cli11/`

Libraries are header-only... will be automatically included during the build process (hopefully).
