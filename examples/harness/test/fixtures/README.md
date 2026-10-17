These images were drawn with a Chrome canvas: 1200 × 360 pixels, white background,
black 48px sans-serif text at (50, 110) and (50, 230):

```text
Verification code: 583721
확인 번호: 583721
```

`text.png` and `text.jpeg` contain these pixels. `blank.png` has the same white
background with no text. CPU tests use the images directly, so CI needs no browser
or installed fonts. The browser integration test draws fresh images with random
codes that are absent from the agent's prompt and file names.
