import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./index.css";

import App from "./App.tsx";


// ============================================================================
// INITIAL THEME
// ============================================================================

function initializeTheme() {
  const saved =
    localStorage.getItem(
      "rag-theme",
    );

  if (saved === "light") {
    document.documentElement.classList.add(
      "light",
    );

    return;
  }

  if (saved === "dark") {
    document.documentElement.classList.remove(
      "light",
    );

    return;
  }

  const prefersLight =
    window.matchMedia(
      "(prefers-color-scheme: light)",
    ).matches;

  if (prefersLight) {
    document.documentElement.classList.add(
      "light",
    );
  } else {
    document.documentElement.classList.remove(
      "light",
    );
  }
}

initializeTheme();


// ============================================================================
// RENDER
// ============================================================================

const root =
  document.getElementById(
    "root",
  );

if (!root) {
  throw new Error(
    "Root element was not found.",
  );
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);