import type { CSSProperties, ReactNode } from "react"

interface RowProps {
  align?: CSSProperties["alignItems"]
  gap?: string
  marginTop?: string
  children: ReactNode
}

export function Row({ align = "center", gap = "2rem", marginTop, children }: RowProps) {
  return (
    <div style={{ display: "flex", gap, alignItems: align, flex: 1, minHeight: 0, marginTop }}>
      {children}
    </div>
  )
}

interface ColProps {
  flex?: string
  className?: string
  fontSize?: string
  children: ReactNode
}

export function Col({ flex = "1 1 50%", className, fontSize, children }: ColProps) {
  return (
    <div className={className} style={{ flex, minWidth: 0, fontSize }}>
      {children}
    </div>
  )
}
