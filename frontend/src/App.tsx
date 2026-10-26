import { Routes, Route } from "react-router-dom";
import Shell from "./components/layout/Shell";
import Overview from "./features/overview/Overview";
import ModelRegistry from "./features/models/ModelRegistry";
import ForecastExplorer from "./features/forecast/ForecastExplorer";
import Comparison from "./features/comparison/Comparison";
import TrainingMonitor from "./features/training/TrainingMonitor";
import EvaluationMetrics from "./features/evaluation/EvaluationMetrics";
import Experiments from "./features/experiments/Experiments";
import ExportPage from "./features/export/ExportPage";

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/models" element={<ModelRegistry />} />
        <Route path="/forecast" element={<ForecastExplorer />} />
        <Route path="/comparison" element={<Comparison />} />
        <Route path="/training" element={<TrainingMonitor />} />
        <Route path="/evaluation" element={<EvaluationMetrics />} />
        <Route path="/experiments" element={<Experiments />} />
        <Route path="/export" element={<ExportPage />} />
      </Routes>
    </Shell>
  );
}