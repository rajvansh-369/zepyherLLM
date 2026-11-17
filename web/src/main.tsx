import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// Fonts are bundled with the app, not fetched from a CDN: the page makes no
// request to anyone but its own server, and works offline.
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";

import App from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
