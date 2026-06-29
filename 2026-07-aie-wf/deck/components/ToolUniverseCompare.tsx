const KITCHEN_SINK = [
  'awk', 'sed', 'gcc', 'g++', 'clang', 'less', 'more', 'grep', 'egrep', 'fgrep',
  'make', 'cmake', 'bash', 'zsh', 'sh', 'fish', 'curl', 'wget',
  'tar', 'gzip', 'bzip2', 'xz', 'zip', 'unzip', 'find', 'xargs', 'locate',
  'vim', 'nano', 'emacs', 'ed', 'python3', 'pip', 'pipx', 'node', 'npm', 'pnpm',
  'yarn', 'git', 'svn', 'hg', 'ssh', 'scp', 'rsync', 'cron', 'systemd', 'apt',
  'apt-get', 'dpkg', 'yum', 'dnf', 'brew', 'jq', 'yq', 'tr', 'cut', 'sort',
  'uniq', 'head', 'tail', 'tee', 'cat', 'tac', 'ls', 'cp', 'mv', 'rm', 'mkdir',
  'rmdir', 'chmod', 'chown', 'du', 'df', 'ps', 'top', 'htop', 'kill', 'pkill',
  'nc', 'netstat', 'ss', 'dig', 'nslookup', 'ping', 'traceroute', 'mtr',
  'openssl', 'gpg', 'sha256sum', 'md5sum', 'docker', 'podman', 'kubectl',
  'helm', 'ffmpeg', 'imagemagick', 'convert', 'mogrify', 'pandoc', 'tex',
  'latex', 'tmux', 'screen', 'watch', 'time', 'env', 'printenv', 'history',
  'alias', 'which', 'whereis', 'man', 'info', 'apropos', 'whoami', 'id',
  'sudo', 'su', 'passwd', 'useradd', 'groupadd', 'mount', 'umount', 'lsof',
  'strace', 'ltrace', 'perf', 'gdb', 'valgrind', 'objdump', 'readelf',
];

const CURATED = ['sql_query', 'create_chart', 'create_table', 'load_skill'];

const card: React.CSSProperties = {
  borderRadius: '12px',
  padding: '1rem 1.2rem',
  display: 'flex',
  flexDirection: 'column',
  height: '24rem',
  overflow: 'hidden',
};

const title: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '0.7rem',
  letterSpacing: '0.14em',
  textTransform: 'uppercase',
  marginBottom: '0.7rem',
};

const tag: React.CSSProperties = {
  padding: '0.5rem 1rem',
  borderRadius: '8px',
  fontFamily: 'var(--font-mono)',
  fontSize: '1.15rem',
  background: 'rgba(74, 215, 197, 0.12)',
  color: 'var(--accent-aqua)',
  border: '1px solid rgba(74, 215, 197, 0.35)',
};

// deterministic pseudo-random from index
function rand(i: number, salt: number) {
  const x = Math.sin(i * 9301 + salt * 49297) * 233280;
  return x - Math.floor(x);
}

export default function ToolUniverseCompare() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.2rem', marginTop: '0.4rem' }}>
      <div
        style={{
          ...card,
          border: '1px solid var(--accent-aqua)',
          background: 'rgba(74, 215, 197, 0.06)',
        }}
      >
        <div style={{ ...title, color: 'var(--accent-aqua)' }}>The sandbox</div>
        <div
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-evenly',
            alignItems: 'center',
            gap: '0.6rem',
          }}
        >
          {CURATED.map((t) => (
            <span key={t} style={tag}>
              {t}
            </span>
          ))}
        </div>
      </div>

      <div
        style={{
          ...card,
          border: '1px solid rgba(255, 138, 76, 0.4)',
          background: 'rgba(255, 138, 76, 0.04)',
          position: 'relative',
        }}
      >
        <div style={{ ...title, color: 'var(--accent-secondary)' }}>The desert</div>
        <div
          style={{
            flex: 1,
            lineHeight: 1.0,
            fontFamily: 'var(--font-mono)',
            overflow: 'hidden',
          }}
        >
          {KITCHEN_SINK.map((t, i) => {
            const size = 0.9 + rand(i, 1) * 1.1;
            const opacity = 0.4 + rand(i, 2) * 0.6;
            const skew = (rand(i, 3) - 0.5) * 8;
            return (
              <span
                key={t}
                style={{
                  display: 'inline-block',
                  margin: '0.1rem 0.45rem',
                  fontSize: `${size}rem`,
                  opacity,
                  color: i % 5 === 0 ? 'var(--accent-secondary)' : 'var(--color-text)',
                  transform: `rotate(${skew}deg)`,
                }}
              >
                {t}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
