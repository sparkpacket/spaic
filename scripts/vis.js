// Basic visualization: draws network JSON (neataptic) as nodes and weighted edges.
// Accepts neatJSON (network.toJSON()) and draws into canvasID.

function drawNetworkOnCanvas(canvas, neatJSON) {
  if (!neatJSON) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  // extract nodes and connections if present
  const nodes = (neatJSON.nodes || []);
  const conns = (neatJSON.connections || []);
  // Simple layout: inputs left, outputs right, hidden spread in middle
  const inputs = nodes.filter(n => n.type === 'input');
  const outputs = nodes.filter(n => n.type === 'output');
  const hidden = nodes.filter(n => n.type === 'hidden');

  const padding = 40;
  const leftX = padding;
  const rightX = canvas.width - padding;
  const midX = (canvas.width/2);
  // vertical spacing
  function place(list, x) {
    const step = canvas.height / (list.length + 1);
    list.forEach((n,i)=>{
      n._x = x;
      n._y = (i+1)*step;
    });
  }
  place(inputs, leftX);
  place(hidden, midX);
  place(outputs, rightX);

  // draw connections
  conns.forEach(c=>{
    const from = nodes.find(n=>n.id === c.from);
    const to = nodes.find(n=>n.id === c.to);
    if (!from || !to) return;
    const w = c.weight || 0;
    ctx.beginPath();
    ctx.moveTo(from._x, from._y);
    ctx.lineTo(to._x, to._y);
    ctx.lineWidth = 1 + Math.min(6, Math.abs(w)*6);
    ctx.strokeStyle = w>0 ? 'rgba(110,231,183,0.9)' : 'rgba(231,110,147,0.9)';
    ctx.stroke();
  });

  // draw nodes
  nodes.forEach(n=>{
    ctx.beginPath();
    ctx.fillStyle = n.type === 'input' ? 'rgba(120,180,255,0.9)' : (n.type === 'output' ? 'rgba(255,200,100,0.9)' : 'rgba(200,200,200,0.9)');
    ctx.arc(n._x, n._y, 8, 0, Math.PI*2);
    ctx.fill();
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.font = '10px monospace';
    ctx.fillText(String(n.id), n._x+10, n._y+4);
  });
}
