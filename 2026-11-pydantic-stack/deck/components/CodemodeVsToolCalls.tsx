import type { CSSProperties } from "react";

/** Side-by-side comparison: one LLM round-trip per tool call vs. one run_code call that makes them all. */

const TOOL_CALLS = ["get_weather('New York')", "get_weather('London')", "get_weather('Tokyo')"];

const wrap: CSSProperties = {
	display: "grid",
	gridTemplateColumns: "1fr 1fr",
	gap: "0.9rem",
	fontFamily: "var(--font-mono)",
	alignItems: "stretch",
};

const column: CSSProperties = {
	display: "flex",
	flexDirection: "column",
	alignItems: "stretch",
	gap: "0.2rem",
};

const colTitle: CSSProperties = {
	fontSize: "0.65rem",
	letterSpacing: "0.14em",
	textTransform: "uppercase",
	color: "var(--color-muted)",
	textAlign: "center",
	marginBottom: "0.4rem",
};

const llm: CSSProperties = {
	border: "1px solid var(--accent)",
	borderRadius: "8px",
	padding: "0.28rem 0.4rem",
	background: "color-mix(in srgb, var(--accent) 14%, transparent)",
	color: "var(--color-heading)",
	fontWeight: 600,
	fontSize: "0.72rem",
	textAlign: "center",
};

const tool: CSSProperties = {
	border: "1px solid rgba(255,255,255,0.14)",
	borderRadius: "6px",
	padding: "0.22rem 0.3rem",
	background: "rgba(255,255,255,0.03)",
	color: "var(--color-text)",
	fontSize: "0.56rem",
	textAlign: "center",
	whiteSpace: "nowrap",
};

const arrow: CSSProperties = {
	color: "var(--color-muted)",
	fontSize: "0.7rem",
	textAlign: "center",
	lineHeight: 1,
	userSelect: "none",
};

const sandbox: CSSProperties = {
	flex: 1,
	display: "flex",
	flexDirection: "column",
	justifyContent: "center",
	gap: "0.35rem",
	border: "1px solid var(--accent-aqua)",
	borderRadius: "10px",
	padding: "0.5rem 0.45rem",
	background: "color-mix(in srgb, var(--accent-aqua) 9%, transparent)",
	boxShadow: "0 0 0 3px color-mix(in srgb, var(--accent-aqua) 8%, transparent)",
};

const sandboxTitle: CSSProperties = {
	color: "var(--accent-aqua)",
	fontWeight: 600,
	fontSize: "0.72rem",
	textAlign: "center",
};

const sandboxNote: CSSProperties = {
	color: "var(--color-muted)",
	fontSize: "0.56rem",
	textAlign: "center",
};

const total: CSSProperties = {
	marginTop: "0.3rem",
	textAlign: "center",
	fontSize: "0.64rem",
	color: "var(--color-muted)",
	lineHeight: 1.3,
};

const totalNumber: CSSProperties = {
	display: "block",
	fontSize: "1.2rem",
	fontWeight: 700,
	lineHeight: 1.1,
};

export default function CodemodeVsToolCalls() {
	return (
		<div style={wrap}>
			<div style={column}>
				<div style={colTitle}>Tool calls</div>
				{TOOL_CALLS.map((call) => (
					<div key={call} style={column}>
						<div style={llm}>LLM</div>
						<div style={arrow}>↓</div>
						<div style={tool}>{call}</div>
						<div style={arrow}>↓</div>
					</div>
				))}
				<div style={llm}>LLM</div>
				<div style={total}>
					<span style={{ ...totalNumber, color: "var(--accent)" }}>{TOOL_CALLS.length + 1}</span>
					LLM calls, every result
					<br />
					in the context window
				</div>
			</div>

			<div style={column}>
				<div style={colTitle}>Code mode</div>
				<div style={llm}>LLM</div>
				<div style={arrow}>↓</div>
				<div style={sandbox}>
					<div style={sandboxTitle}>run_code</div>
					{TOOL_CALLS.map((call) => (
						<div key={call} style={tool}>
							{call}
						</div>
					))}
					<div style={sandboxNote}>loops, branches and state stay in the sandbox</div>
				</div>
				<div style={arrow}>↓</div>
				<div style={llm}>LLM</div>
				<div style={total}>
					<span style={{ ...totalNumber, color: "var(--accent-aqua)" }}>2</span>
					LLM calls, however
					<br />
					many tool calls
				</div>
			</div>
		</div>
	);
}
