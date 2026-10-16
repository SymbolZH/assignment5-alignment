
(()=>{'use strict';
for(const button of document.querySelectorAll('.copy-code')){
 button.addEventListener('click',async()=>{
  const code=button.closest('.code-block').querySelector('code');
  const text=code.dataset.source;
  try{
   if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(text);
   else{
    const selection=window.getSelection();const range=document.createRange();range.selectNodeContents(code);selection.removeAllRanges();selection.addRange(range);
    if(!document.execCommand('copy'))throw new Error('copy unavailable');
    selection.removeAllRanges();
   }
   button.textContent='已复制';setTimeout(()=>button.textContent='复制代码',1500);
  }catch(_){button.textContent='请选中代码复制';setTimeout(()=>button.textContent='复制代码',2500);}
 });
}
const taskIds=new Set([...document.querySelectorAll('.problem-heading')].map(h=>h.id));
function markTarget(){for(const h of document.querySelectorAll('.problem-heading'))h.classList.remove('target-heading');const id=decodeURIComponent(location.hash.slice(1));if(taskIds.has(id)){const h=document.getElementById(id);h.classList.add('target-heading');const back=document.querySelector('[data-back-link]');if(back&&location.pathname.indexOf('safety-supplement')===-1)back.href='assignment5-requirements.html#'+id;}}
markTarget();window.addEventListener('hashchange',markTarget);
})();
