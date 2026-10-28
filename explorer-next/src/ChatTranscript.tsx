import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { Button } from './components/ui/button'
import { Badge } from './components/ui/badge'
import { Message,MessageContent,MessageFooter } from './components/ui/message'
import { Bubble,BubbleContent } from './components/ui/bubble'
import { MessageScrollerProvider,MessageScroller,MessageScrollerViewport,MessageScrollerContent,MessageScrollerItem,MessageScrollerButton } from './components/ui/message-scroller'
import { chatTurns,setupChanges } from './chat-turns'
import { columnLabel,type Chat } from './storage'
import { grade } from './prompts'
export function ChatTranscript({chat,busy,rerun}:{chat:Chat;busy:boolean;rerun:()=>void}){
 const turns=chatTurns(chat),columns=chat.columns.filter(c=>!c.hidden),style={gridTemplateColumns:`repeat(${columns.length},minmax(0,1fr))`}
 return <MessageScrollerProvider><MessageScroller><MessageScrollerViewport aria-label="Chat messages"><MessageScrollerContent className="turn-list">{turns.length?turns.map((turn,index)=>{
 const changes=setupChanges(turns[index-1]?.setup,turn.setup)
 return <MessageScrollerItem key={turn.key} messageId={`${chat.id}-${turn.key}`} scrollAnchor><div className="chat-turn">
 {changes.length>0&&<div className="setup-divider">{changes.map((change,i)=><span key={i} className={change.meaning?`meaning-${change.meaning}`:undefined}>{change.text}</span>)}</div>}
 <div className="turn-columns" style={style}>{columns.map(c=><div className="turn-messages" data-column={c.id} key={c.id}>{c.messages.filter((m,i)=>(m.id??`legacy-${m.role==='user'?i:i-1}`)===turn.key).map((m,i)=><Message key={i} align={m.role==='user'?'end':'start'}><MessageContent><Bubble variant={m.role==='user'?'secondary':'ghost'}><BubbleContent className={m.role==='assistant'?'markdown':'whitespace-pre-wrap'}>{m.role==='assistant'&&m.content?<Markdown skipHtml remarkPlugins={[remarkGfm,remarkMath]} rehypePlugins={[[rehypeKatex,{throwOnError:false,strict:false}]]} components={{table:({children})=><div className="markdown-table" tabIndex={0}><table>{children}</table></div>}}>{m.content}</Markdown>:m.content||(busy?'Thinking…':'No reply')}</BubbleContent></Bubble>{m.role==='assistant'&&!busy&&(g=>g&&<MessageFooter><Badge variant="outline" className={g.correct?'check-ok':m.stopped==='length'?undefined:'check-bad'}>{g.correct?`✓ ${g.answer}`:m.stopped==='length'?`cut off before the answer (${g.answer})`:`✗ expected ${g.answer}`}</Badge></MessageFooter>)(grade(turn.text,m.content))}{((m.stopped&&['loop','client','interrupted'].includes(m.stopped))||(!busy&&m.role==='assistant'&&!m.stopped))&&<MessageFooter><Badge variant="outline">stopped: {m.stopped||'interrupted'}</Badge></MessageFooter>}</MessageContent></Message>)}</div>)}</div>
 {index===turns.length-1&&<div className="turn-actions"><Button variant="ghost" size="sm" disabled={busy} onClick={rerun}>Rerun</Button></div>}
 </div></MessageScrollerItem>
 }):<div className="turn-columns" style={style}>{columns.map(c=><p className="empty" key={c.id}>{c.id==='normal'?'Original model':'Neuron push '+columnLabel(c,chat)}</p>)}</div>}</MessageScrollerContent></MessageScrollerViewport><MessageScrollerButton/></MessageScroller></MessageScrollerProvider>
}
