// Inline stroke icons; decorative unless a caller gives the surrounding control a name.

type IconProps = { className?: string };

function Svg({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <svg className={className ?? "icon"} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      {children}
    </svg>
  );
}

export const IconSearch = ({ className }: IconProps) => (
  <Svg className={className}>
    <circle cx="8.5" cy="8.5" r="5.5" />
    <path d="M12.6 12.6 17 17" />
  </Svg>
);

export const IconZoomIn = ({ className }: IconProps) => (
  <Svg className={className}>
    <circle cx="8.5" cy="8.5" r="5.5" />
    <path d="M12.6 12.6 17 17M6 8.5h5M8.5 6v5" />
  </Svg>
);

export const IconZoomOut = ({ className }: IconProps) => (
  <Svg className={className}>
    <circle cx="8.5" cy="8.5" r="5.5" />
    <path d="M12.6 12.6 17 17M6 8.5h5" />
  </Svg>
);

export const IconCheck = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M4 10.5 8 14.5 16 5.5" />
  </Svg>
);

export const IconChevronDown = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="m5 8 5 5 5-5" />
  </Svg>
);

export const IconChevronUp = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="m5 12 5-5 5 5" />
  </Svg>
);

export const IconChevronLeft = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="m12 5-5 5 5 5" />
  </Svg>
);

export const IconChevronRight = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="m8 5 5 5-5 5" />
  </Svg>
);

export const IconArrowUp = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M10 16V4M5 9l5-5 5 5" />
  </Svg>
);

export const IconArrowDown = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M10 4v12M5 11l5 5 5-5" />
  </Svg>
);

export const IconUndo = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M6 3 3 6l3 3M3 6h8.5a5 5 0 0 1 0 10H7" />
  </Svg>
);

export const IconClose = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="m5 5 10 10M15 5 5 15" />
  </Svg>
);

export const IconWarning = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M10 2.5 18 16.5H2ZM10 8v3.5M10 14h0" />
  </Svg>
);

export const IconInfo = ({ className }: IconProps) => (
  <Svg className={className}>
    <circle cx="10" cy="10" r="7.5" />
    <path d="M10 9v5M10 6.5h0" />
  </Svg>
);

export const IconDownload = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M10 3v10M5.5 8.5 10 13l4.5-4.5M3.5 16.5h13" />
  </Svg>
);

export const IconUpload = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M10 14V3M5.5 7.5 10 3l4.5 4.5M3.5 16.5h13" />
  </Svg>
);

export const IconCopy = ({ className }: IconProps) => (
  <Svg className={className}>
    <rect x="7" y="7" width="10" height="10" rx="2" />
    <path d="M13 7V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2" />
  </Svg>
);

export const IconOffline = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M2.5 7.5c4.3-3.6 10.7-3.6 15 0M5.5 11c2.5-2 6.5-2 9 0M10 15.5h0" />
  </Svg>
);

export const IconSpinner = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M10 3a7 7 0 1 0 7 7" />
  </Svg>
);

export const IconMenu = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M3 5.5h14M3 10h14M3 14.5h14" />
  </Svg>
);

export const IconLink = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M8.5 11.5a3.5 3.5 0 0 0 5 0l2.5-2.5a3.5 3.5 0 0 0-5-5l-1 1" />
    <path d="M11.5 8.5a3.5 3.5 0 0 0-5 0L4 11a3.5 3.5 0 0 0 5 5l1-1" />
  </Svg>
);

export const IconShared = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M4 10.5 8 14.5 16 5.5" />
  </Svg>
);

export const IconDiffer = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M3 6.5h14M3 13.5h14M6 3.5 3 6.5l3 3M14 10.5l3 3-3 3" />
  </Svg>
);

export const IconOneSide = ({ className }: IconProps) => (
  <Svg className={className}>
    <path d="M3 10h6M11 5h6v10h-6Z" />
  </Svg>
);

export const IconQuestion = ({ className }: IconProps) => (
  <Svg className={className}>
    <circle cx="10" cy="10" r="7.5" />
    <path d="M8 7.8a2.2 2.2 0 1 1 3 2c-.6.3-1 .8-1 1.4M10 14h0" />
  </Svg>
);

export function SideMarker({ side, size = "1rem" }: { side: "source" | "target"; size?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" focusable="false" className="side-marker">
      {side === "source" ? <circle cx="8" cy="8" r="6.5" fill="var(--source)" /> : <rect x="1.5" y="1.5" width="13" height="13" rx="3" fill="var(--target)" />}
    </svg>
  );
}
