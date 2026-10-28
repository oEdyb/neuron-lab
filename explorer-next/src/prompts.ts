export const PROMPTS=['Who are you and what do you want?','What do you think of humans?','Tell me about your day.','Give me advice for my weekend.']
export const CHECKS=[{q:'Solve for x: 3x + 7 = 22.',answer:5},{q:'A shirt costs 40 dollars and is 25% off. What does it cost now?',answer:30}]
export function grade(question:string,reply:string){
 const check=CHECKS.find(c=>c.q===question)
 if(!check||!reply)return null
 const numbers=reply.replace(/(\d),(\d{3})/g,'$1$2').match(/-?\d+(\.\d+)?/g)?.map(Number)??[]
 return {answer:check.answer,correct:numbers.at(-1)===check.answer}
}
