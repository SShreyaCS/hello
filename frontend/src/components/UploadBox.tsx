import { ChangeEvent, useRef, DragEvent } from "react";
import { UploadCloud } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type UploadBoxProps = {
  onUpload: (file: File) => void;
  onReset: () => void;
  previewUrl?: string | null;
  imageName?: string | null;
  hasResults: boolean;
  isAnalyzing: boolean;
};

const UploadBox = ({
  onUpload,
  onReset,
  previewUrl,
  imageName,
  hasResults,
  isAnalyzing
}: UploadBoxProps) => {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onUpload(file);
      event.target.value = "";
    }
  };

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      onUpload(file);
    }
  };

  const preventDefaults = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
  };

  return (
    <div className="space-y-4">
      <label
        htmlFor="retina-upload"
        onDrop={handleDrop}
        onDragOver={preventDefaults}
        onDragEnter={preventDefaults}
        onDragLeave={preventDefaults}
        className={cn(
          "border-2 border-dashed border-primary/40 rounded-3xl p-6 md:p-10",
          "bg-slate-900/60 hover:border-primary hover:bg-slate-900/80",
          "transition-colors flex flex-col items-center space-y-4",
          isAnalyzing && "pointer-events-none opacity-70",
          !previewUrl && "cursor-pointer"
        )}
        onClick={() => {
          if (!previewUrl && !isAnalyzing) {
            inputRef.current?.click();
          }
        }}
      >
        {previewUrl ? (
          <div className="w-full space-y-3 text-center">
            <div className="w-full rounded-2xl overflow-hidden border border-white/10 bg-black/40">
              <img
                src={previewUrl}
                alt={imageName ?? "Uploaded retina image"}
                className="w-full h-64 object-contain bg-black"
              />
            </div>
            {imageName && (
              <p className="text-xs text-muted-foreground truncate">{imageName}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Drop a new file to analyze again
            </p>
          </div>
        ) : (
          <>
            <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center">
              <UploadCloud className="h-10 w-10 text-primary" />
            </div>
            <div className="text-center space-y-2">
              <p className="text-lg font-semibold text-slate-100">
                Drag & drop your eye image
              </p>
              <p className="text-sm text-muted-foreground">
                or click to browse from your device (PNG / JPG)
              </p>
            </div>
          </>
        )}
        <input
          id="retina-upload"
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
      </label>

      <Button
        onClick={() => {
          if (hasResults) {
            onReset();
            setTimeout(() => inputRef.current?.click(), 80);
          } else {
            inputRef.current?.click();
          }
        }}
        disabled={isAnalyzing}
        className={cn(
          "w-full bg-gradient-to-r from-primary to-accent shadow-lg",
          hasResults && "from-primary/10 to-accent/10 text-primary border border-primary/40 bg-transparent"
        )}
      >
        {hasResults ? "Re-scan" : "Upload Image"}
      </Button>
    </div>
  );
};

export default UploadBox;

