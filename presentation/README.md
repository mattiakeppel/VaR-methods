# Presentation

Source of the 24-slide presentation *Value at Risk (VaR) Forecasting and Diagnostic Validation*, built with [Manim](https://github.com/ManimCommunity/manim) and [Manim Slides](https://github.com/jeertmans/manim-slides).

## Files

- `presentation.py`: all slides, as a single Manim Slides scene
- `pit.py`: PIT and Q–Q animations (a simulated example and the crisis-period FHS-GJR forecasts)
- `theme.py`: colours, typography and layout helpers
- `notes.json`: speaker notes for each slide
- `build.py`: renders the deck and exports the HTML and PDF
- `data/forecasts.npz`: daily returns, FHS-GJR VaR, the FHS-GJR forecast distributions (support and resample counts) and the PIT values for Normal, HS and FHS-GJR, 2013–2021
- `data/backtests.json`: Kupiec, Christoffersen and conditional-coverage p-values per period, model and confidence level
- `assets/`: portfolio table, Latin Modern font (GUST font licence), GitHub/ChatGPT marks and the repository QR code

The returns, VaR, PIT values and backtest results in `data/` match the output of `../main.m` (`output/tables/`).

## Build

Requires Python, a LaTeX installation (e.g. TeX Live) and the packages in `requirements.txt`.

```
pip install -r requirements.txt
python build.py
```

This writes:

- `final_presentation.html`: a single self-contained file that plays offline in a browser. Use the arrow keys to move, **F** for full screen and **S** for speaker notes.
- `final_presentation.pdf`: the final frame of every numbered slide, with the two simulated Q–Q slides merged into one page and a vector closing page with clickable links.

Rendering takes a while. For a quick, low-quality preview, run `manim render -ql presentation.py Presentation`, then `manim-slides present Presentation`.
