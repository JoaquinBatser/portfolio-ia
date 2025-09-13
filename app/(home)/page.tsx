import Link from "next/link";

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col justify-center px-4 mt-8">
      {/* Hero Section */}
      <div className="mx-auto max-w-4xl text-center">
        <h1 className="mb-6 text-4xl font-bold tracking-tight sm:text-6xl">
          Desarrollando competencias en{" "}
          <span className="bg-gradient-to-r from-slate-700 to-slate-900 dark:from-slate-300 dark:to-slate-100 bg-clip-text text-transparent">
            Inteligencia Artificial
          </span>{" "}
          y Ciencia de Datos
        </h1>

        <p className="mb-8 text-lg text-fd-muted-foreground sm:text-xl">
          Portfolio académico documentando mi aprendizaje en análisis exploratorio de datos,
          visualización avanzada y deployment de aplicaciones modernas.
        </p>

        <div className="mb-12 flex flex-col gap-4 sm:flex-row sm:justify-center">
          <Link
            href="/docs"
            className="rounded-lg bg-fd-primary px-6 py-3 text-fd-primary-foreground font-semibold hover:bg-fd-primary/90 transition-colors"
          >
            Explorar Documentación
          </Link>

        </div>

        {/* Process Steps */}
        <div className="mb-16 grid gap-8 md:grid-cols-3">
          <div className="text-center">
            <div className="mb-4 mx-auto flex h-12 w-12 items-center justify-center rounded-lg bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300">
              <span className="text-xl font-bold">1</span>
            </div>
            <h3 className="mb-2 text-lg font-semibold">Explorar</h3>
            <p className="text-fd-muted-foreground">
              Análisis exploratorio de datasets reales como Netflix y taxis de NYC
            </p>
          </div>

          <div className="text-center">
            <div className="mb-4 mx-auto flex h-12 w-12 items-center justify-center rounded-lg bg-slate-200 text-slate-800 dark:bg-slate-700 dark:text-slate-200">
              <span className="text-xl font-bold">2</span>
            </div>
            <h3 className="mb-2 text-lg font-semibold">Visualizar</h3>
            <p className="text-fd-muted-foreground">
              Creación de dashboards interactivos con matplotlib, seaborn y pandas
            </p>
          </div>

          <div className="text-center">
            <div className="mb-4 mx-auto flex h-12 w-12 items-center justify-center rounded-lg bg-slate-300 text-slate-900 dark:bg-slate-600 dark:text-slate-100">
              <span className="text-xl font-bold">3</span>
            </div>
            <h3 className="mb-2 text-lg font-semibold">Implementar</h3>
            <p className="text-fd-muted-foreground">
              Deployment profesional con Next.js, Fumadocs y Vercel
            </p>
          </div>
        </div>

        {/* Featured Projects */}
        <div className="mb-16">
          <h2 className="mb-8 text-2xl font-bold">Proyectos Destacados</h2>
          <div className="grid gap-6 md:grid-cols-2">
            <div className="rounded-lg border border-fd-border p-6 text-left hover:shadow-md transition-shadow">
              <h3 className="mb-2 text-lg font-semibold">Análisis Netflix Dataset</h3>
              <p className="mb-3 text-fd-muted-foreground">
                EDA completo del catálogo de Netflix con visualizaciones avanzadas
              </p>
              <Link
                href="/docs/EDA%20y%20Fuentes/03-EDA-Netflix-Dataset-con-pandas"
                className="text-fd-primary hover:underline font-medium"
              >
                Ver análisis →
              </Link>
            </div>

            <div className="rounded-lg border border-fd-border p-6 text-left hover:shadow-md transition-shadow">
              <h3 className="mb-2 text-lg font-semibold">EDA Multi-fuentes NYC</h3>
              <p className="mb-3 text-fd-muted-foreground">
                Integración de 3M+ registros de taxis con técnicas avanzadas de joins
              </p>
              <Link
                href="/docs/EDA%20y%20Fuentes/04-EDA-Multi-fuentes-y-Joins"
                className="text-fd-primary hover:underline font-medium"
              >
                Ver proyecto →
              </Link>
            </div>
          </div>
        </div>

        {/* Tech Stack */}
        <div className="mb-16">
          <h2 className="mb-8 text-2xl font-bold">Stack Tecnológico</h2>
          <div className="flex flex-wrap justify-center gap-4">
            {["Python", "pandas", "matplotlib", "seaborn", "Next.js", "TypeScript", "Vercel", "Fumadocs"].map((tech) => (
              <span
                key={tech}
                className="rounded-full bg-fd-accent px-4 py-2 text-sm font-medium"
              >
                {tech}
              </span>
            ))}
          </div>
        </div>

        {/* CTA */}
        <div className="rounded-lg bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-900/50 dark:to-slate-800/50 border border-slate-200 dark:border-slate-700 p-8">
          <h2 className="mb-4 text-2xl font-bold">Portfolio en Evolución</h2>
          <p className="mb-6 text-fd-muted-foreground">
            Esta documentación refleja mi crecimiento académico y profesional en IA y Ciencia de Datos,
            desde conceptos básicos hasta proyectos complejos con datos a escala.
          </p>
          <Link
            href="/docs"
            className="inline-flex items-center rounded-lg bg-slate-900 dark:bg-slate-100 px-6 py-3 text-slate-100 dark:text-slate-900 font-semibold hover:bg-slate-800 dark:hover:bg-slate-200 transition-colors"
          >
            Comenzar Exploración
          </Link>
        </div>
      </div>
    </main>
  );
}
