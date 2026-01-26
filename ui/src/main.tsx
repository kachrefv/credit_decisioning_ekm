import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HeroUIProvider } from "@heroui/react";
import { ThemeProvider } from './contexts/ThemeContext.tsx'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HeroUIProvider>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </HeroUIProvider>
  </StrictMode>,
)
