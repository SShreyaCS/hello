import { ReactNode } from "react";
import { cn } from "@/lib/utils";

type CardProps = {
  children: ReactNode;
  className?: string;
};

export const Card = ({ children, className }: CardProps) => {
  return (
    <div
      className={cn(
        "rounded-3xl bg-slate-900/70 border border-white/5 backdrop-blur-sm",
        "shadow-xl shadow-black/40",
        className
      )}
    >
      {children}
    </div>
  );
};

