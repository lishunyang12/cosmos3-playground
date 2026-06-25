// Lucide-style line icons (stroke = currentColor). Replaces emoji-as-icons,
// per the UI/UX rule "no emoji icons — use SVG". Sized in em so they inherit
// the surrounding font-size; color inherits via currentColor.
const S = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24"
    fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round"
    strokeLinejoin="round" aria-hidden="true" focusable="false" {...props}
  />
);

export const IconAtom = (p) => (
  <S {...p}>
    <circle cx="12" cy="12" r="1" />
    <path d="M20.2 20.2c2.04-2.03.02-7.36-4.5-11.9-4.54-4.52-9.87-6.54-11.9-4.5-2.04 2.03-.02 7.36 4.5 11.9 4.54 4.52 9.87 6.54 11.9 4.5Z" />
    <path d="M15.7 15.7c4.52-4.54 6.54-9.87 4.5-11.9-2.03-2.04-7.36-.02-11.9 4.5-4.52 4.54-6.54 9.87-4.5 11.9 2.03 2.04 7.36.02 11.9-4.5Z" />
  </S>
);

export const IconSparkles = (p) => (
  <S {...p}>
    <path d="M9.94 15.5A2 2 0 0 0 8.5 14.06l-6.14-1.58a.5.5 0 0 1 0-.96L8.5 9.94A2 2 0 0 0 9.94 8.5l1.58-6.14a.5.5 0 0 1 .96 0L14.06 8.5A2 2 0 0 0 15.5 9.94l6.14 1.58a.5.5 0 0 1 0 .96L15.5 14.06a2 2 0 0 0-1.44 1.44l-1.58 6.14a.5.5 0 0 1-.96 0Z" />
    <path d="M20 3v4" /><path d="M22 5h-4" />
  </S>
);

export const IconEye = (p) => (
  <S {...p}>
    <path d="M2.06 12.35a1 1 0 0 1 0-.7 10.75 10.75 0 0 1 19.88 0 1 1 0 0 1 0 .7 10.75 10.75 0 0 1-19.88 0" />
    <circle cx="12" cy="12" r="3" />
  </S>
);

export const IconUpload = (p) => (
  <S {...p}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" /><line x1="12" x2="12" y1="3" y2="15" />
  </S>
);

export const IconAlert = (p) => (
  <S {...p}>
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
    <path d="M12 9v4" /><path d="M12 17h.01" />
  </S>
);

export const IconChevron = (p) => (<S {...p}><path d="m9 18 6-6-6-6" /></S>);

export const IconReset = (p) => (
  <S {...p}>
    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
    <path d="M3 3v5h5" />
  </S>
);

export const IconDownload = (p) => (
  <S {...p}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="7 10 12 15 17 10" /><line x1="12" x2="12" y1="15" y2="3" />
  </S>
);
