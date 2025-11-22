import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Landing from './pages/Landing'
import UploadPage from './pages/UploadPage'
import EdgePage from './pages/EdgePage'
import HistoryPage from './pages/HistoryPage'
import SettingsPage from './pages/SettingsPage'
import AboutPage from './pages/AboutPage'
import './App.css'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/edge" element={<EdgePage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </Router>
  )
}

export default App
