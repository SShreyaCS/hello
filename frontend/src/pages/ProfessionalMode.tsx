import { useState } from "react";
import { ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import Layout from "@/components/Layout";
import UploadBox from "@/components/UploadBox";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
type DiseaseResult = {
  label: string;
  name: string;
  confidence: number;
  probability: number;
  severity: string;
  threshold: number;
  detected?: boolean;
};

type PredictionResponse = {
  top_prediction: DiseaseResult;
  predictions: DiseaseResult[];
  detected_diseases: DiseaseResult[];
  recommendation: string;
  explanation?: string;
  heatmap_image?: string;
  all_probabilities?: Record<string, number>;
};

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:5001";

const ProfessionalMode = () => {
  const navigate = useNavigate();
  const [results, setResults] = useState<PredictionResponse | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedImageName, setSelectedImageName] = useState<string | null>(null);

  const analyzeFile = async (file: File) => {
    setError(null);
    setAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const message = await response.json().catch(() => ({}));
        throw new Error(message.detail ?? "Failed to analyze image.");
      }

      const data: PredictionResponse = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleUpload = (file: File) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const dataUrl = reader.result as string;
      setPreviewUrl(dataUrl);
      setSelectedImageName(file.name);
      setResults(null);
      setError(null);
      analyzeFile(file);
    };
    reader.onerror = () => {
      setError("Unable to read the selected image. Please try another file.");
    };
    reader.readAsDataURL(file);
  };

  const handleReset = () => {
    setResults(null);
    setError(null);
    setAnalyzing(false);
    setPreviewUrl(null);
    setSelectedImageName(null);
  };

  return (
    <Layout>
      <div className="min-h-screen py-8">
        <div className="max-w-5xl mx-auto space-y-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 animate-fade-in">
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => navigate("/home")}
                className="hover:bg-accent/10"
              >
                <ArrowLeft className="h-5 w-5" />
              </Button>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  Generic Eye Analysis
                </h1>
                <p className="text-muted-foreground text-sm">
                  Generic disease scan diagnosis
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              onClick={() => navigate("/generic")}
              className="self-start sm:self-auto hover:bg-accent/10"
            >
              Switch to Professional Mode
            </Button>
          </div>

          <div className="grid md:grid-cols-2 gap-6 lg:gap-8">
            <div className="space-y-4 animate-slide-up">
              <h2 className="text-xl font-semibold">Upload External Eye Image</h2>
              <UploadBox
                onUpload={handleUpload}
                onReset={handleReset}
                previewUrl={previewUrl}
                imageName={selectedImageName}
                hasResults={Boolean(results)}
                isAnalyzing={analyzing}
              />
              {analyzing && (
                <Card className="p-6 border-2 border-primary/20 shadow-lg">
                  <div className="space-y-3">
                    <p className="text-sm font-medium">
                      Analyzing image. Please wait...
                    </p>
                    <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                      <div className="h-full w-1/3 bg-gradient-to-r from-primary/30 via-primary to-primary/30 animate-progress" />
                    </div>
                  </div>
                </Card>
              )}
              {error && (
                <Card className="p-4 border border-red-500/40 bg-red-500/10 text-red-200">
                  <p className="text-sm">{error}</p>
                </Card>
              )}
              {results?.heatmap_image && (
                <Card className="p-4 border-2 border-primary/20 shadow-lg">
                  <h3 className="text-sm font-semibold mb-2">Grad-CAM Visualization</h3>
                  <div className="rounded-lg overflow-hidden border border-border">
                    <img
                      src={results.heatmap_image}
                      alt="Grad-CAM heatmap overlay"
                      className="w-full h-auto"
                    />
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Heat map showing model attention for: {results.top_prediction.name}
                  </p>
                </Card>
              )}
            </div>

            <div
              className="space-y-4 animate-slide-up"
              style={{ animationDelay: "100ms" }}
            >
              <h2 className="text-xl font-semibold">Analysis Results</h2>
              {results ? (
                <Card className="p-6 space-y-6 border-2 hover:border-primary/20 transition-all shadow-lg">
                  <div className="space-y-4">
                    <div>
                      <p className="text-xs uppercase tracking-widest text-muted-foreground">
                        Top finding
                      </p>
                      <h3 className="text-2xl font-semibold text-primary">
                        {results.top_prediction.name}
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        Confidence:{" "}
                        <span className="font-medium">
                          {results.top_prediction.confidence.toFixed(1)}%
                        </span>
                      </p>
                    </div>

                    <h3 className="font-semibold text-lg">
                      Detected Conditions
                    </h3>
                    {(results.detected_diseases &&
                    results.detected_diseases.length > 0
                      ? results.detected_diseases
                      : [results.top_prediction]
                    ).map((disease) => (
                      <div
                        key={disease.label}
                        className="space-y-2 p-4 rounded-lg bg-accent/5 border border-white/5"
                      >
                        <div className="flex justify-between items-center">
                          <span className="font-medium">{disease.name}</span>
                          <span className="text-xs text-muted-foreground">
                            {disease.label}
                          </span>
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className="text-muted-foreground">
                              Confidence
                            </span>
                            <span className="font-medium">
                              {disease.confidence.toFixed(1)}%
                            </span>
                          </div>
                          <Progress value={disease.confidence} className="h-2" />
                          <div className="flex justify-between text-xs text-muted-foreground">
                            <span>Probability</span>
                            <span>{(disease.probability * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {results.explanation && (
                    <div className="pt-4 border-t border-border">
                      <h4 className="font-semibold mb-2">Explanation</h4>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {results.explanation}
                      </p>
                    </div>
                  )}

                  <div className="pt-4 border-t border-border">
                    <h4 className="font-semibold mb-2">Recommendation</h4>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {results.recommendation}
                    </p>
                  </div>

                </Card>
              ) : (
                <Card className="p-12 text-center border-2 border-dashed hover:border-primary/30 transition-all">
                  <p className="text-muted-foreground">
                    Upload an image or scan live to see results
                  </p>
                </Card>
              )}
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default ProfessionalMode;

