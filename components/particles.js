const canvas = document.getElementById("particleCanvas");
    const ctx = canvas.getContext("2d");

    let particles = [];
    let animationFrameId;
    let mouseX = 0;
    let mouseY = 0;

    // Resize canvas and generate particles
    function resizeCanvas() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      createParticles();
    }

    // Initialize particles with random position and velocity
    function createParticles() {
        const numParticles = (canvas.width * canvas.height) / 6000;
      particles = Array.from({ length: numParticles }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 3 + 1,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        alpha: Math.random() * 0.5 + 0.5,
      }));
    }

    // Draw a line between two particles based on distance
    function drawLine(p1, p2, distance, maxDistance) {
      const alpha = 1 - distance / maxDistance;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.strokeStyle = `rgba(56, 189, 248, ${alpha * 0.3})`; // Light blue glow
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Main animation loop
    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((p, index) => {
        // Move particle
        p.x += p.vx;
        p.y += p.vy;

        // Bounce off edges
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        // Mouse repulsion
        const dx = mouseX - p.x;
        const dy = mouseY - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          const force = (120 - dist) / 120;
          p.x -= dx * force * 0.02;
          p.y -= dy * force * 0.02;
        }

        // Draw particle
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(56, 189, 248, ${p.alpha * 0.6})`;
        ctx.fill();

        // Connect nearby particles
        for (let j = index + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx2 = p.x - p2.x;
          const dy2 = p.y - p2.y;
          const distance = Math.sqrt(dx2 * dx2 + dy2 * dy2);
          if (distance < 100) drawLine(p, p2, distance, 100);
        }
      });

      animationFrameId = requestAnimationFrame(animate);
    }

    // Track mouse movement
    function handleMouseMove(e) {
      mouseX = e.clientX;
      mouseY = e.clientY;
    }

    // Event listeners
    window.addEventListener("resize", resizeCanvas);
    window.addEventListener("mousemove", handleMouseMove);

    // Initialize
    resizeCanvas();
    animate();