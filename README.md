# Conway Screensaver (Rust / wgpu)



A high-performance, GPU-accelerated Cellular Automata engine. Originally prototyped in Python, this version has been rewritten in **Rust** using `wgpu` (WebGPU) for rendering and `rayon` for parallel simulation. It features an "Autopilot" that creates living textures, a decoupled physics/render loop, and a runtime debug overlay.

## 🚀 Quick Start

### 1. Build
You need the [Rust Toolchain](https://rustup.rs/) installed.

```bash
# Build optimized release binary
cargo build --release
```

### 2. Locate Output
The executable will be generated here:
```
./target/release/conway_screensaver.exe
```

### 3. Run
* **Windowed Mode:** `cargo run --release`
* **Debug Mode (Overlay on):** `cargo run --release -- --debug`
* **Screensaver Mode:** `cargo run --release -- /s`

---

## 🖥️ How to Install as a Windows Screensaver

Windows screensavers are simply executables with a `.scr` extension that accept specific command line arguments (`/s`, `/c`, `/p`).

1.  **Build the release binary** (see above).
2.  Navigate to `./target/release/`.
3.  Copy `conway_screensaver.exe` to a safe location (e.g., your Pictures folder).
4.  **Rename** the file from `.exe` to `.scr`.
5.  Right-click the `.scr` file and select **Install**.
6.  Windows will open the Screen Saver Settings dialog with "Conway_screensaver" selected.

---

## 🎮 Controls & Hotkeys

| Input | Function |
| :--- | :--- |
| **Esc** | Exit the screensaver. |
| **D** | Toggle **Debug Overlay** (Stats + Flash Text). |
| **Right Arrow** | Switch to the **Next** variant. |
| **Left Arrow** | Switch to the **Previous** variant. |
| **Up Arrow** | **Reseed** the current variant (new random noise). |
| **Mouse Wheel** | Increase / Decrease User Brush Size. |
| **LMB (Hold)** | Paint "Alive" cells. |
| **RMB (Hold)** | Erase / Paint "Dead" cells. |
| **+** / **=** | Increase user paint density. |
| **-** / **_** | Decrease user paint density. |

---

## 🧪 Simulation Variants Explained

The engine supports several mathematical models. The autopilot will cycle through these automatically, or you can switch manually.

### Life-Like (Binary)
These operate on a standard grid where cells are 0 (Dead) or 1 (Alive).
* **Standard (Conway):** The Game of Life. Balanced chaos.
* **HighLife:** Similar to Conway, but adds a "Replicator" rule (Born on 6 neighbors).
* **Day & Night:** A symmetric rule where dead cells inside a blob behave exactly like live cells in empty space.
* **Seeds:** Explodes violently. Very chaotic, low survival rate.

### Multi-State / History
* **Immigration:** Two warring species (colors). New cells take the color of the majority parent.
* **QuadLife:** Four distinct species. When a cell is born from mixed parents, the new color is determined by majority or a specific decay logic.
* **Generations:** Cells don't die immediately. They become "old" (refractory state) and fade out over several frames. This creates trailing tails behind moving patterns.
* **Brian's Brain:** A three-state system (Firing -> Refractory -> Off). Mimics neural networks and creates spaceship-like structures.

### Alternative Neighborhoods
* **Larger Than Life:** Instead of looking at 8 immediate neighbors, cells count neighbors in a wide radius (e.g., range 5). Behaves like a microscopic oil-water simulation.
* **Margolus:** Uses a block-partition grid (2x2 blocks) that shifts and rotates. Excellent for simulating gas particles or sand-like physics.

### Continuous / Math
* **Cyclic:** A "Rock-Paper-Scissors" war. State 1 eats 0, 2 eats 1, etc. Creates swirling spirals and demon-like patterns.
* **Gray-Scott:** A Reaction-Diffusion simulation using continuous floating-point math ($u, v$), mapped to a color palette. Looks like biological coral growth or fingerprints.

---

## 🎛️ Fine-Tuning (The Control Panel)

You don't need to touch the raw simulation code to change the "personality" of the screensaver. All behavioral logic is centralized in:

**File:** `src/tuning.rs`

Look for the `ControlPanel::for_sim` function. Here you can adjust "Knobs" that are defined as `AnimatedF32` (values that drift over time) or static ranges.

### Key Knobs to Tweak

1.  **Speed of Time:**
    ```rust
    sim_steps_per_second: AnimatedF32 { ... range: RangeF32 { min: 5.0, max: 20.0 } ... }
    ```
    * Increase `max` to make the simulation run faster.

2.  **Autopilot Boredom:**
    ```rust
    reset_after_seconds: 60.0,
    ```
    * How long before the system wipes the screen and picks a new rule.

3.  **Colors:**
    In `palette.accent` and `palette.background`:
    * Change `initial` to set the starting color.
    * Change `retarget` range to make colors shift faster or slower.

4.  **Autopilot Painting:**
    * **Blobs:** Large circles of random noise. Tweak `blobs_per_second`.
    * **Cursors:** Invisible brushes that wander the screen.
        * `xy_retarget`: How erratic the movement is.
        * `stamp_60hz`: How "thick" the paint is.

---

## 🐍 Legacy Python Prototype

This project started as a Python script using `pygame`. While easier to prototype, it struggled with performance at 4K resolutions and was difficult to distribute as a standalone `.exe`.

**Original Requirements:**
* `numpy` (Calculation)
* `pygame` (Rendering)
* `scipy` (Convolution for Larger Than Life)
* `psutil` / `pynvml` (Debug stats)

**Why Rust?**
1.  **Performance:** Rust's `rayon` allows multithreaded simulation steps, and `wgpu` blits to the screen with zero cost.
2.  **Safety:** No more random crashes after 4 hours of runtime due to memory leaks.
3.  **Distribution:** Compiles to a single, static `.exe` file (~2MB) with no dependencies.

---

## 🛠️ Technical Details

* **Render Engine:** `wgpu` (WebGPU). Uses a texture upload strategy where the CPU calculates the grid, writes to a mapped buffer, and the GPU simply displays it.
* **Simulation:** Running on the CPU using `rayon` `par_chunks_mut`.
* **Timing:**
    * **Render Loop:** VSync capped (usually 60 or 144Hz).
    * **Sim Loop:** Decoupled. Accumulates delta time and steps at the target IPS (Iterations Per Second).
    * **Paint Loop:** Fixed 60Hz tick for cursor movement to ensure smooth drawing curves regardless of simulation speed.