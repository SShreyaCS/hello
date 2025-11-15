import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import ProfessionalMode from "./pages/ProfessionalMode";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ProfessionalMode />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;

