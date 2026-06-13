import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

// Initialize the GoogleGenAI SDK using the recommended static configuration
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    }
  }
});

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Body parser
  app.use(express.json());

  // API endpoint for generating Career Advisor report
  app.post("/api/generate-report", async (req, res) => {
    try {
      const studentData = req.body;
      if (!studentData || !studentData.fullName || !studentData.preferredIndustry) {
        return res.status(400).json({ error: "Missing required student profile data." });
      }

      console.log(`[Advisor Backend] Processing report request for: ${studentData.fullName}`);

      const userPrompt = `You are an expert career counselor.

Analyze the student's profile:
- Student Name: ${studentData.fullName}
- Current Education/Degree: ${studentData.currentEducation}
- Current Skills & Proficiencies: ${studentData.currentSkills}
- Interests & Hobbies: ${studentData.interests}
- Personal Strengths: ${studentData.strengths}
- Development Needs / Weaknesses: ${studentData.weaknesses}
- Preferred Industry target: ${studentData.preferredIndustry}
- Ultimate Career Goal: ${studentData.careerGoal}

Generate matching and realistic recommendations for them. Formulate professional advice covering all 9 required areas detailed in the JSON schema.`;

      // Define structured response schema matching the requested items precisely
      const carrierSchema = {
        type: Type.OBJECT,
        properties: {
          careerRecommendation: {
            type: Type.OBJECT,
            properties: {
              title: { type: Type.STRING, description: "Highly specific best career path recommended for the student" },
              summary: { type: Type.STRING, description: "A detailed professional description of why this recommendation suits the student" }
            },
            required: ["title", "summary"]
          },
          whySuits: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: "At least 3 analytical reasons/observations detailing why this path is an absolute match for their strengths, hobbies, and goals."
          },
          skillsGapAnalysis: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                skill: { type: Type.STRING, description: "The specific skill name they must learn (e.g. Docker, Deep Learning, Public Speaking)" },
                gap: { type: Type.STRING, description: "What exactly is key to learn and why they are lacking it now" },
                priority: { type: Type.STRING, description: "Must be 'High', 'Medium', or 'Low'" }
              },
              required: ["skill", "gap", "priority"]
            },
            description: "Critical skills analysis comparing their current skills to target profile requirements"
          },
          roadmap: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                month: { type: Type.STRING, description: "Logical phase duration, e.g., 'Month 1', 'Month 2', 'Month 3-4', 'Month 5-6'" },
                focus: { type: Type.STRING, description: "Core technical topic or theme focus for this period" },
                topics: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                  description: "Key skills, concepts or tools to learn in this block"
                },
                actionSteps: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                  description: "Concrete hands-on projects, exercises, or certificates they should build/earn"
                }
              },
              required: ["month", "focus", "topics", "actionSteps"]
            },
            description: "An actionable, highly progressive 6-Month Learning Roadmap divided into logical chapters"
          },
          recommendedCourses: {
            type: Type.OBJECT,
            properties: {
              free: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    title: { type: Type.STRING, description: "Plausible/real free course name" },
                    platform: { type: Type.STRING, description: "e.g., Coursera, YouTube, freeCodeCamp, edX" }
                  },
                  required: ["title", "platform"]
                }
              },
              paid: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    title: { type: Type.STRING, description: "Plausible/real professional paid course or certificate title" },
                    platform: { type: Type.STRING, description: "e.g., Udemy, Coursera Plus, Udacity NanoDegree, LinkedIn Learning" }
                  },
                  required: ["title", "platform"]
                }
              }
            },
            required: ["free", "paid"]
          },
          salaryRange: {
            type: Type.OBJECT,
            properties: {
              min: { type: Type.STRING, description: "Realistic initial/starting annual salary range in Indian Rupees (INR) for the Indian job market, strictly matching actual entry-level industry averages in India, formatted with the ₹ symbol (e.g., '₹4,50,000 - ₹6,00,000')" },
              max: { type: Type.STRING, description: "Experienced or lead professional annual salary range in Indian Rupees (INR) for the Indian job market, formatted with the ₹ symbol (e.g., '₹15,00,000 - ₹25,00,000')" },
              average: { type: Type.STRING, description: "Industry standard median annual salary in Indian Rupees (INR), formatted with the ₹ symbol (e.g., '₹8,50,000')" }
            },
            required: ["min", "max", "average"]
          },
          industryTrends: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: "At least 3 emerging and hot market developments, tools, or paradigm shifts reshaping this target landscape"
          },
          motivationMessage: { type: Type.STRING, description: "An inspiring, highly personalized motivational signing-off message from you" }
        },
        required: [
          "careerRecommendation",
          "whySuits",
          "skillsGapAnalysis",
          "roadmap",
          "recommendedCourses",
          "salaryRange",
          "industryTrends",
          "motivationMessage"
        ]
      };

      // Helper function to query Gemini with instant cascade fallback to avoid delay/timeout
      async function queryGeminiWithRetryAndFallback(userPrompt: string, carrierSchema: any) {
        const modelsToTry = ["gemini-3.5-flash", "gemini-flash-latest", "gemini-3.1-flash-lite"];
        let lastError: any = null;

        for (const modelName of modelsToTry) {
          try {
            console.log(`[Advisor Backend] Querying model ${modelName}...`);
            const start = Date.now();
            const response = await ai.models.generateContent({
              model: modelName,
              contents: userPrompt,
              config: {
                systemInstruction: "You are an expert career counselor with 20+ years of experiences helping college students navigate tech and business fields. Give incredibly insightful, customized advice based strictly on the provided profile. All salary data MUST be provided in Indian Rupees (INR) matching realistic, non-exaggerated Indian job market standards. Ensure your salary structures are humble, correct, and represent true entry-level and experienced standards in India (using the ₹ symbol, formatted standardly like ₹5,50,000).",
                responseMimeType: "application/json",
                responseSchema: carrierSchema,
                temperature: 0.4,
              }
            });

            const text = response.text;
            if (!text) {
              throw new Error("Empty response received from the GenAI model.");
            }
            console.log(`[Advisor Backend] Success with model ${modelName} in ${Date.now() - start}ms.`);
            return JSON.parse(text);
          } catch (error: any) {
            lastError = error;
            console.warn(`[Advisor Backend Warning] Model ${modelName} was congested or failed:`, error?.message || error);
            // Cascade instantly to the next model in the priority list without adding sleeping delays
          }
        }
        throw lastError || new Error("Failed to generate report after trying all available fallback models.");
      }

      // Query Gemini using our robust helper
      const parsedReport = await queryGeminiWithRetryAndFallback(userPrompt, carrierSchema);
      return res.json(parsedReport);

    } catch (error: any) {
      console.error("[Advisor Backend Error] Failed to generate career report:", error);
      return res.status(500).json({
        error: error?.message || "Internal GenAI error while synthesizing recommendation"
      });
    }
  });

  // Setup Dev vs Production static file serving with Vite middleware logic
  if (process.env.NODE_ENV !== "production") {
    console.log("[Express] Starting in Development mode, mounting Vite Dev Middleware...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("[Express] Starting in Production mode, serving static output files...");
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Express Server] Running on http://localhost:${PORT}`);
  });
}

startServer();
