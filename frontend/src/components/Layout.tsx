import { ReactNode } from "react";
import { cn } from "@/lib/utils";

type LayoutProps = {
  children: ReactNode;
  className?: string;
};

const Layout = ({ children, className }: LayoutProps) => {
  return (
    <div className={cn("min-h-screen bg-slate-950 text-slate-100", className)}>
      <main className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">
        {children}
      </main>
    </div>
  );
};

export default Layout;

