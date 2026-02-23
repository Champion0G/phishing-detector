"use client";

import { useState } from "react";

interface ScanResult {
  url: string;
  result: string;
  probability: number;
  reasons: string[];
  note?: string;
  stats: {
    length: number;
    entropy: number;
    is_tech_tld: boolean;
  };
}

export default function Home() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleScan = async () => {
    if (!url) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) throw new Error("Connection failed");
      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to scan URL");
    } finally {
      setLoading(false);
    }
  };

  const isPhishing = result?.result?.includes("PHISHING");

  return (
    <main className="min-h-screen relative bg-glow flex flex-col items-center">
      {/* Navbar Minimalist */}
      <nav className="w-full max-w-5xl py-6 px-8 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-blue-600 rounded-md flex items-center justify-center font-black text-[10px]">PS</div>
          <span className="font-bold tracking-tight text-sm">Phishing Shield AI</span>
        </div>
        <div className="flex gap-4 text-xs font-medium text-secondary">
          <span className="hover:text-white cursor-pointer transition-colors">V14 Hybrid</span>
          <span className="text-border">|</span>
          <span className="hover:text-white cursor-pointer transition-colors">Docs</span>
        </div>
      </nav>

      <div className="w-full max-w-3xl px-6 pt-20 space-y-12">
        <header className="text-center space-y-4">
          <h2 className="text-3xl font-semibold tracking-tight">Protecting your digital navigation.</h2>
          <p className="text-secondary text-sm max-w-lg mx-auto leading-relaxed">
            Enter any URL to perform a 128-point hybrid security audit using our character-sentiment ML engine.
          </p>
        </header>

        {/* Command Bar / Input Area */}
        <section className="cursor-card p-1.5 flex flex-col md:flex-row gap-2">
          <div className="flex-1 flex items-center px-4 py-2">
            <svg className="w-4 h-4 text-secondary mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
            </svg>
            <input
              type="text"
              placeholder="Paste URL to scan..."
              className="w-full bg-transparent border-none outline-none text-sm placeholder:text-neutral-700"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleScan()}
            />
          </div>
          <button
            onClick={handleScan}
            disabled={loading || !url}
            className="cursor-btn cursor-btn-primary text-sm min-w-[120px]"
          >
            {loading ? "Analyzing..." : "Run Analysis"}
          </button>
        </section>

        {error && (
          <div className="animate-dropdown p-4 cursor-card border-red-500/20 bg-red-500/5 flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-red-500" />
            <p className="text-red-400 text-xs font-medium">{error}. Is the backend running?</p>
          </div>
        )}

        {result && (
          <div className="animate-dropdown space-y-6">
            <div className={`cursor-card border-t-2 ${isPhishing ? 'border-t-red-500' : 'border-t-blue-500'} p-0 overflow-hidden`}>
              {/* Header */}
              <div className="px-8 py-6 border-b border-border flex justify-between items-center">
                <div className="space-y-1">
                  <p className="text-[10px] font-black uppercase tracking-widest text-secondary">Detection Result</p>
                  <h3 className={`text-2xl font-bold ${isPhishing ? 'text-red-500' : 'text-blue-500'}`}>
                    {result.result}
                  </h3>
                </div>
                <div className="text-right space-y-1">
                  <p className="text-[10px] font-black uppercase tracking-widest text-secondary">Confidence</p>
                  <p className="font-mono text-xl">{result.probability}%</p>
                </div>
              </div>

              {/* Analysis Details */}
              <div className="p-8 space-y-8">
                {result.reasons && result.reasons.length > 0 && (
                  <div className="space-y-4">
                    <h4 className="text-[10px] font-black uppercase tracking-widest text-secondary">Analysis Notes</h4>
                    <div className="space-y-2">
                      {result.reasons.map((reason, i) => (
                        <div key={i} className="flex items-center gap-3 text-sm text-neutral-300">
                          <div className={`w-1 h-3 rounded-full ${isPhishing ? 'bg-red-500' : 'bg-blue-500'}`} />
                          {reason}
                        </div>
                      ))}
                      {result.note && (
                        <div className="flex items-center gap-3 text-xs text-secondary mt-2 italic">
                          • {result.note}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Grid Stats */}
                <div className="grid grid-cols-3 gap-6 pt-6 border-t border-border">
                  <div className="space-y-1">
                    <p className="text-[9px] font-black uppercase text-secondary">Domain Entropy</p>
                    <p className="text-sm font-mono">{result.stats.entropy}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-[9px] font-black uppercase text-secondary">Char Length</p>
                    <p className="text-sm font-mono">{result.stats.length}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-[9px] font-black uppercase text-secondary">Structure</p>
                    <p className="text-sm font-mono text-blue-500">{result.stats.is_tech_tld ? "Professional" : "Standard"}</p>
                  </div>
                </div>
              </div>
            </div>

            <p className="text-center text-[10px] text-neutral-600 font-mono">
              Fingerprint: {Math.random().toString(16).slice(2, 10).toUpperCase()} • V14-STABLE
            </p>
          </div>
        )}

        {!result && !loading && !error && (
          <div className="grid grid-cols-2 gap-4 mt-12">
            {[
              { label: "Typosquatting", desc: "Checks for lookalive characters like 'goog1e'" },
              { label: "TLD Neutrality", desc: "Balanced trust for .io, .dev, and .app" },
              { label: "Identity Check", desc: "Reputation anchor for major tech brands" },
              { label: "Character-Sentiment", desc: "LGBM analysis on ngram sequences" }
            ].map((feature, i) => (
              <div key={i} className="p-4 cursor-card opacity-50 hover:opacity-100 transition-opacity space-y-1">
                <p className="text-xs font-bold text-blue-500">{feature.label}</p>
                <p className="text-[10px] text-secondary leading-tight">{feature.desc}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
