import { ButtonHTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "outline" | "ghost";
  size?: "default" | "sm" | "lg" | "icon";
};

const variantClasses: Record<
  NonNullable<ButtonProps["variant"]>,
  string
> = {
  default:
    "bg-primary text-white hover:bg-primary/90 shadow-md shadow-primary/20",
  outline:
    "border border-primary/40 text-primary hover:bg-primary/10 hover:border-primary/60",
  ghost: "bg-transparent hover:bg-primary/10 text-primary"
};

const sizeClasses: Record<NonNullable<ButtonProps["size"]>, string> = {
  default: "px-4 py-2 h-11 rounded-xl text-sm font-medium",
  sm: "px-3 py-1.5 h-9 rounded-lg text-sm",
  lg: "px-5 py-3 h-12 rounded-xl text-base font-medium",
  icon: "h-10 w-10 rounded-xl flex items-center justify-center"
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = "default",
      size = "default",
      type = "button",
      ...props
    },
    ref
  ) => (
    <button
      ref={ref}
      type={type}
      className={cn(
        "transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
        variantClasses[variant],
        sizeClasses[size],
        className
      )}
      {...props}
    />
  )
);

Button.displayName = "Button";

