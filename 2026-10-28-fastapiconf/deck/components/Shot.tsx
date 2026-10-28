/** A screenshot, or a labelled placeholder until the image is dropped into assets/. */
export default function Shot({ src, label, height = '15rem' }: { src?: string; label: string; height?: string }) {
  if (src) {
    return <img src={src} alt={label} style={{ display: 'block', alignSelf: 'center', margin: '0 auto', width: 'auto', height: 'auto', maxWidth: '100%', maxHeight: height, objectFit: 'contain', borderRadius: 10, border: '1px solid rgba(255,255,255,0.12)' }} />;
  }
  return (
    <div
      style={{
        height,
        borderRadius: 10,
        border: '2px dashed rgba(255,255,255,0.2)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        padding: '1rem',
        color: 'var(--color-muted)',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.85rem',
      }}
    >
      screenshot: {label}
    </div>
  );
}
