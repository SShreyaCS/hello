import { cn } from "@/lib/utils";

type ProgressProps = {
  value: number;
  className?: string;
};

export const Progress = ({ value, className }: ProgressProps) => {
  return (
    <div
      className={cn(
        "w-full h-2 rounded-full bg-slate-800/80 overflow-hidden",
        className
      )}
    >
      <div
        className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-500"
        style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
      />
    </div>
  );
};

