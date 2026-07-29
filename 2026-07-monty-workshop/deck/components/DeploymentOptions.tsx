const AQUA = "#4ad7c5"; // --accent-aqua
const TERTIARY = "#b388ff"; // --accent-tertiary
const SECONDARY = "#ff8a4c"; // --accent-secondary

type Option = {
	title: string;
	color: string;
	bullets: string[];
	footerLabel: string;
	footerValue: string;
};

const OPTIONS: Option[] = [
	{
		title: "Monty OSS",
		color: AQUA,
		bullets: ["Runs in your own process pool", "Application sandbox only", "Free"],
		footerLabel: "latency",
		footerValue: "~5µs",
	},
	{
		title: "Monty over websocket",
		color: TERTIARY,
		bullets: ["Run local or SaaS", "Application + container security", "Paid"],
		footerLabel: "latency",
		footerValue: "~50µs",
	},
	{
		title: "Monty wire to sandbox",
		color: SECONDARY,
		bullets: ["Full CPython", "External sandbox provider", "Dependencies, bash etc.", "Paid"],
		footerLabel: "latency",
		footerValue: "~1s",
	},
];

function Box({ title, color, bullets, footerLabel, footerValue }: Option) {
	return (
		<div
			style={{
				display: "flex",
				flexDirection: "column",
				border: `2px solid ${color}`,
				borderRadius: "18px",
				background: `${color}14`,
				padding: "1.1rem 1.4rem",
				height: "100%",
				boxSizing: "border-box",
			}}
		>
			<div
				style={{
					fontSize: "1.5rem",
					fontWeight: 700,
					color,
					textAlign: "center",
					marginBottom: "0.8rem",
				}}
			>
				{title}
			</div>

			<ul
				style={{
					flex: 1,
					margin: 0,
					paddingLeft: "1.3rem",
					fontSize: "1.25rem",
					lineHeight: 1.7,
				}}
			>
				{bullets.map((b) => (
					<li key={b}>{b}</li>
				))}
			</ul>

			<div
				style={{
					marginTop: "0.9rem",
					paddingTop: "0.7rem",
					borderTop: `1px solid ${color}44`,
				}}
			>
				<div
					style={{
						fontSize: "0.78rem",
						letterSpacing: "0.12em",
						textTransform: "uppercase",
						color: "var(--color-muted)",
					}}
				>
					{footerLabel}
				</div>
				<div
					style={{
						fontFamily: "var(--font-mono)",
						fontSize: "1.5rem",
						fontWeight: 700,
						color,
					}}
				>
					{footerValue}
				</div>
			</div>
		</div>
	);
}

export default function DeploymentOptions() {
	return (
		<div
			style={{
				display: "grid",
				gridTemplateColumns: "1fr 1fr 1fr",
				gap: "2rem",
				marginTop: "1rem",
			}}
		>
			{OPTIONS.map((o) => (
				<Box key={o.title} {...o} />
			))}
		</div>
	);
}
