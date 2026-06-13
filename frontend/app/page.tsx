import Link from "next/link";

export default function LandingPage() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-6 bg-warm-50">
      <div className="max-w-xl text-center">
        <h1 className="text-4xl sm:text-5xl font-semibold text-sage-700 mb-4">
          MannSaathi
        </h1>
        <p className="text-lg text-gray-700 mb-2">
          Feeling overwhelmed? Talk it out. We&apos;ll listen.
        </p>
        <p className="text-sm text-gray-500 mb-10">
          A calm, private space — not therapy, but a starting point.
        </p>

        <Link
          href="/chat"
          className="inline-block bg-sage-500 hover:bg-sage-700 transition-colors text-white px-8 py-3 rounded-full text-lg font-medium"
        >
          Start talking
        </Link>

        <div className="mt-12 text-xs text-gray-500 leading-relaxed">
          MannSaathi is an AI companion, not a therapist.
          <br />
          In crisis? Call <strong>iCall: 9152987821</strong> (free,
          confidential, 24/7).
        </div>
      </div>
    </main>
  );
}
