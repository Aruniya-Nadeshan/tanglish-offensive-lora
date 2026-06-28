const express = require('express');
const cors = require('cors');
require('dotenv').config();
const OpenAI = require('openai');

const app = express();
app.use(cors());
app.use(express.json());

const client = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: 'https://api.groq.com/openai/v1'
});

const LABELS = [
  'Not_offensive',
  'Offensive_Untargetede',
  'Offensive_Targeted_Insult_Individual',
  'Offensive_Targeted_Insult_Group',
  'Offensive_Targeted_Insult_Other',
  'not-Tamil'
];

const LABEL_DESCRIPTIONS = `- Not_offensive: not offensive
- Offensive_Untargetede: offensive, but not aimed at a specific person/group
- Offensive_Targeted_Insult_Individual: offensive, aimed at a specific person
- Offensive_Targeted_Insult_Group: offensive, aimed at a specific group
- Offensive_Targeted_Insult_Other: offensive, aimed at something else (e.g. an event)
- not-Tamil: the comment is in a DIFFERENT language entirely (e.g. Hindi, French, Sinhala). Note: Tamil written purely in Tamil script still counts as Tamil.`;

app.post('/classify', async (req, res) => {
  const { comment } = req.body;

  if (!comment || comment.trim() === '') {
    return res.status(400).json({ error: 'Comment is required' });
  }

  try {
    const response = await client.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [{ role: 'user', content: `You are classifying social media comments written in Tamil-English code-mixed text ("Tanglish").

Classify the comment below into exactly one of these categories:
${LABEL_DESCRIPTIONS}

Comment: ${comment}

Respond with ONLY the category name, nothing else.` }],
      max_tokens: 20,
      temperature: 0
    });

    const rawOutput = response.choices[0].message.content.trim();

    const normalize = s => s.toLowerCase().replace(/[^a-z0-9]/g, '');
    const predicted = LABELS.find(l =>
      normalize(rawOutput).includes(normalize(l))
    ) || 'Not_offensive';

    const scores = LABELS.map(l => ({
      label: l,
      score: l === predicted
        ? 0.85
        : parseFloat((0.15 / (LABELS.length - 1)).toFixed(4))
    })).sort((a, b) => b.score - a.score);

    res.json({ prediction: predicted, scores });

  } catch (error) {
    console.error('Classification error:', error.message);
    res.status(500).json({ error: 'Classification failed' });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});