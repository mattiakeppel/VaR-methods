"""The full 24-slide deck as a single Manim Slides scene."""
import json
from html import escape
import numpy as np
from scipy.stats import norm,t
from manim import *
from manim_slides import Slide
from manim.mobject.svg.svg_mobject import SVG_HASH_TO_MOB_MAP
from theme import *
from pit import PITAnimation,crisis_mask,forecast_scores,qq_cloud,qq_infinity_rows,simulated_pit,QQ_X_LIMIT,QQ_Y_LIMIT

DATA=json.loads((ROOT/"data/backtests.json").read_text(encoding="utf-8"))
EMP=np.load(ROOT/"data/forecasts.npz")
NOTES=json.loads((ROOT/"notes.json").read_text(encoding="utf-8"))
TOTAL=24
AUTHORS=("Angelo Righetti","Elia Mosca","Jessica Tisato","Mattia Keppel","Mattia Maino","Sebastiano Eccher")
REPO_URL="https://github.com/mattiakeppel/VaR-methods"
LINKS={"manim":"https://github.com/ManimCommunity/manim","manim_slides":"https://github.com/jeertmans/manim-slides",
       "project":REPO_URL,"codex":"https://openai.com/codex/"}

def bounds(mob):
    """Link area as percentages of the 16:9 frame, shared by the HTML and PDF exports."""
    return dict(left=(float(mob.get_left()[0])+8)/16*100,top=(4.5-float(mob.get_top()[1])-.08)/9*100,
                width=float(mob.width)/16*100,height=(float(mob.height)+.16)/9*100)

def export_vector(objects,path):
    """Write the closing page as plain SVG paths, so the PDF can carry it as vectors."""
    paths=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080">',
           '<rect width="1920" height="1080" fill="white"/>','<g transform="translate(960 540) scale(120 -120)">']
    for mob in Group(*objects).family_members_with_points():
        if not isinstance(mob,VMobject):continue
        commands=[]
        for subpath in mob.get_subpaths():
            commands.append(f'M {subpath[0][0]:.7f} {subpath[0][1]:.7f}')
            for curve in subpath.reshape((-1,4,3)):
                commands.append('C '+' '.join(f'{p[0]:.7f} {p[1]:.7f}' for p in curve[1:]))
        paths.append(f'<path d="{" ".join(commands)}" fill="{escape(mob.get_fill_color().to_hex())}" '
                     f'fill-opacity="{mob.get_fill_opacity():.5f}" stroke="{escape(mob.get_stroke_color().to_hex())}" '
                     f'stroke-opacity="{mob.get_stroke_opacity():.5f}" stroke-width="{mob.get_stroke_width()*.01:.5f}"/>')
    paths+=['</g>','</svg>']
    path.write_text('\n'.join(paths),encoding='utf-8')

class Presentation(Slide,PITAnimation):
    skip_reversing=True
    max_duration_before_split_reverse=None

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.number=0
        # Copy each frame before writing it, so the video encoder never sees a reused buffer.
        writer=self.renderer.file_writer; original=writer.write_frame
        def owned(frame,num_frames=1):
            original(frame.copy() if isinstance(frame,np.ndarray) else frame,num_frames=num_frames)
        writer.write_frame=owned

    def frame(self,section,title,clear=True):
        """Start a numbered slide: title, top rule and page number."""
        if clear:
            self.clear(); SVG_HASH_TO_MOB_MAP.clear()
        self.number+=1
        print(f"Slide {self.number:02d}/{TOTAL}: {title}",flush=True)
        top=Line([-7.4,3.63,0],[7.4,3.63,0],color=INK,stroke_width=.9)
        heading=left(fit(label(title,34,INK),14.8),-7.4,3.97)
        number=label(f"{self.number:02d} / {TOTAL}",14,MUTED).move_to([7.0,-4.22,0])
        self.chrome=VGroup(top,heading,number)
        if clear:self.add(self.chrome)

    def change_header(self,old):
        """Only one title is painted at a time during connected chart transitions."""
        return Succession(FadeOut(old,run_time=.18),FadeIn(self.chrome,run_time=.22))

    def finish(self,key,hold=1.1):
        """End the current slide and attach its speaker notes."""
        self.wait(hold)
        self.next_slide()
        note=NOTES[key]
        self._slides[-1].notes=f"{self.number:02d}. {note['title']}\n\n{note['notes']}"

    def reveal(self,*objects):
        visible=[o for o in objects if o.get_num_points() or len(o.submobjects)]
        if visible:
            duration=.85 if self.number in {3,4,5,8,14,15,17,22,23} else .65
            self.play(LaggedStart(*[FadeIn(o,shift=UP*.08) for o in visible],lag_ratio=.06),run_time=duration,rate_func=smooth)

    def construct(self):
        self.holding(); self.opening(); self.question(); self.literature()
        self.methods(); self.fhs_theory(); self.fhs_filtering(); self.fhs_limits(); self.pit_theory()
        self.pit_demo(False); self.qq_from_pit(False); self.qq_tails()
        self.data_slide(); self.backtests("stable"); self.return_chart("stable")
        self.backtests("stress"); self.return_chart("stress")
        self.pit_demo(True); self.qq_from_pit(True); self.qq_comparison()
        self.conclusions(); self.limits(); self.closing()

    def holding(self):
        self.clear(); self.number=1; self.camera.background_color=ManimColor(BG)
        self.finish("blank",hold=.35)

    def opening(self):
        self.clear(); self.number+=1
        title=VGroup(label('Value at Risk (VaR) Forecasting',43),label('and Diagnostic Validation',43)).arrange(DOWN,buff=.15).move_to([0,.2,0])
        rule=Line([-5.7,-.75,0],[5.7,-.75,0],color=INK,stroke_width=.8)
        names=VGroup(*[label(name,21).move_to([-4.35+4.35*(i%3),-1.05-.45*(i//3),0]) for i,name in enumerate(AUTHORS)])
        self.play(LaggedStart(*[FadeIn(line,shift=DOWN*.13) for line in title],lag_ratio=.25),run_time=1.6)
        self.play(GrowFromCenter(rule),run_time=1.0)
        self.play(LaggedStart(*[FadeIn(name,shift=DOWN*.065) for name in names],lag_ratio=.15),run_time=1.6)
        self.finish('title',hold=1.2)

    def question(self):
        self.frame("Introduction","Research question")
        question=label("Does Filtered Historical Simulation improve VaR\nforecasting across different market regimes,\nand can we accurately backtest the results?",34)
        fit(question,13.8).move_to([0,1.48,0])
        implication=bullet_block("Financial implications",[
            "Underestimating VaR leads to insufficient capital to cover risks.",
            "Clustered breaches indicate that the model does not respond quickly enough to market stress."],
            14.6,2.15,27).move_to([0,-1.125,0])
        expectations=bullet_block("Expectations",[
            "We expect Filtered Historical Simulation to outperform the other models in backtests."],
            14.6,1.55,27).move_to([0,-3.0,0])
        self.reveal(question,implication,expectations)
        self.finish("question")

    def literature(self):
        self.frame("Introduction","Literature")
        rows=[("Barone-Adesi et al. (1999)","Filtered Historical Simulation","Resample shocks after filtering volatility."),
              ("Diebold, Gunther & Tay (1998)","Probability Integral Transform","Evaluate the entire forecast distribution."),
              ("Kupiec (1995)","Unconditional coverage","Is the exception frequency correct?"),
              ("Christoffersen (1998)","Independence and conditional coverage","Are exceptions clustered?")]
        cards=VGroup(*[block(author,[title+" · "+body],14.4,1.43,23) for author,title,body in rows]).arrange(DOWN,buff=.21).move_to([0,-.33,0])
        for card in cards:self.reveal(card)
        self.finish("literature")

    def methods(self):
        self.frame("Methods","VaR forecasts and backtesting")
        forecasts=bullet_block("1 · Forecast tomorrow's loss threshold",[
            "Variance–covariance: zero-mean Normal law with rolling volatility.",
            "Historical Simulation (HS): empirical distribution of the last 250 returns.",
            "Filtered Historical Simulation (FHS): resampled shocks with GJR-GARCH volatility."],14.4,2.25,24,bold_labels=True).move_to([0,2.15,0])
        convention=panel("2 · Record each exception",[
            eq(r"I_t=\mathbf{1}\{r_t<-\mathrm{VaR}_{c,t}\},\qquad \Pr(I_t=1)=1-c",30),
            label("t: forecast day; r: portfolio return; c: confidence (95% or 99%); VaR: positive loss threshold.",21),
            label("I = 1 for a breach, 0 otherwise; the indicator 1{condition} equals 1 when the condition holds.",21)
        ],height=1.85).move_to([0,-.15,0])
        tests=bullet_block("Test at the 5% significance level (reject if p < 0.05)",[
            "Kupiec / unconditional coverage (UC): correct exception frequency.",
            "Christoffersen / independence (IND): no clustering.",
            "Joint conditional coverage (CC): UC + IND."],14.4,2.15,23,bold_labels=True).move_to([0,-2.33,0])
        self.reveal(forecasts);self.reveal(convention);self.reveal(tests)
        self.finish("methods")

    def fhs_theory(self):
        self.frame("Methods","Filtered Historical Simulation")
        alternatives=bullet_block("Why go beyond the alternatives?",[
            "Parametric (variance–covariance): fast and analytical, but Normality misses fat tails.",
            "Historical Simulation: uses real data and equal weights, but reacts slowly to shocks."],14.4,1.8,24,bold_labels=True).move_to([0,2.30,0])
        innovation=bullet_block("FHS: a semiparametric combination",[
            "Filter historical returns with a conditional volatility model, removing volatility clusters.",
            "Capture leverage with GJR-GARCH: losses can raise volatility more than gains.",
            "Standardise historical shocks, treat them as iid, then reapply next-day volatility."],14.4,2.2,24).move_to([0,.02,0])
        tails=bullet_block("Keep empirical tails; adapt their scale to current risk",[
            "Retain historical extreme events without assuming a Normal return distribution.",
            "Scale past shocks dynamically as market volatility changes."],14.4,1.75,24).move_to([0,-2.14,0])
        process=VGroup()
        for i,(x,text) in enumerate(zip((-5.4,-1.8,1.8,5.4),('Filter volatility','Standardise','Resample','Rescale'))):
            process.add(label(text,23,bold=True).move_to([x,-3.35,0]))
            if i<3:process.add(Arrow([x+1.28,-3.35,0],[x+2.15,-3.35,0],buff=0,color=MUTED,stroke_width=1.1,max_tip_length_to_length_ratio=.12))
        self.reveal(alternatives);self.reveal(innovation);self.reveal(tails)
        self.reveal(process)
        self.finish("fhs_filter")

    def fhs_filtering(self):
        self.frame("Methods","FHS · GJR-GARCH filtering and simulation")
        mean=panel("1 · Assume zero conditional mean",[
            eq(r"\mu_t=\mathbb{E}[r_t\mid\mathcal{F}_{t-1}]=0\quad\Longrightarrow\quad r_t=\mu_t+\epsilon_t=\epsilon_t",32),
            label("Given past information, the expected return is zero. Realised returns still vary.",23)
        ],height=1.6).move_to([0,2.5,0])
        variance=VGroup(left(label('2 · Filter variance: GJR-GARCH(1,1)',25,bold=True),-6.94,1.42),
            Line([-6.94,1.15,0],[6.94,1.15,0],color=RULE,stroke_width=.8),
            left(eq(r"\sigma_t^2=\omega+\beta\sigma_{t-1}^2+\alpha\epsilon_{t-1}^2+\gamma\epsilon_{t-1}^2 I_{t-1}",34),-6.93,.84))
        glossary_left=VGroup(
            eq(r'\sigma_t^2:\ \text{conditional variance}',23),
            eq(r'\epsilon_{t-1}:\ \text{previous return / shock}',23),
            eq(r'I_{t-1}=\mathbf{1}_{\{\epsilon_{t-1}<0\}}:\ \text{negative-shock indicator}',23)
        ).arrange(DOWN,buff=.13,aligned_edge=LEFT)
        glossary_right=VGroup(
            eq(r'\omega:\ \text{intercept};\quad\alpha:\ \text{shock impact}',23),
            eq(r'\beta:\ \text{volatility persistence}',23),
            eq(r'\gamma:\ \text{extra impact of negative shocks}',23)
        ).arrange(DOWN,buff=.13,aligned_edge=LEFT)
        left(glossary_left,-6.93,-.02);left(glossary_right,.40,-.02)
        variance.add(glossary_left,glossary_right,
            left(eq(r'\omega>0,\quad\alpha,\beta\geq0,\quad\alpha+\gamma\geq0;\qquad\alpha+\beta+\frac{\gamma}{2}<1',24),-6.93,-.95),
            left(label('Stationarity condition for symmetric standardised innovations.',20),-6.93,-1.32))
        rescale=VGroup(left(label('3 · Standardise, resample and rescale',25,bold=True),-6.94,-1.90),
            Line([-6.94,-2.16,0],[6.94,-2.16,0],color=RULE,stroke_width=.8),
            left(eq(r'z_t=\frac{r_t}{\sigma_t}=\frac{\epsilon_t}{\sigma_t}',35),-6.93,-2.63),
            left(eq(r'\widetilde r_{t+1}=\widetilde\epsilon_{t+1}=\widetilde\sigma_{t+1}\,\widetilde z',35),.40,-2.63),
            left(label('Resample a historical standardised error; apply next-day conditional volatility.',22),-6.93,-3.23),
            left(label('Tilde denotes a simulated quantity; the simulated shock is the simulated return.',21),-6.93,-3.62))
        self.reveal(mean)
        self.wait(.40)
        self.reveal(variance)
        self.wait(.40)
        self.reveal(rescale)
        self.finish("fhs_theory")

    def fhs_limits(self):
        self.frame("Methods","FHS · Advantages and disadvantages")
        advantages=bullet_block("Advantages",[
            "Responsiveness: today's crash can raise tomorrow's volatility estimate and VaR.",
            "No Normal return-distribution assumption: retain the empirical tails.",
            "Leverage effect: losses can create more uncertainty than gains."],14.4,2.65,25,bold_labels=True,gap=BULLET_ITEM_GAP*1.175).move_to([0,1.79,0])
        disadvantages=bullet_block("Disadvantages",[
            "Long computation times: rolling estimation and resampling.",
            "Historical dependence: a quiet sample can produce overly optimistic paths.",
            ("Extreme stress and outliers can destabilise parameter estimation.","Extreme stress and outliers")],14.4,2.65,25,bold_labels=True,gap=BULLET_ITEM_GAP*1.175).move_to([0,-1.25,0])
        self.reveal(advantages);self.reveal(disadvantages)
        self.reveal(note("Why GJR-GARCH? ARCH/GARCH is symmetric; GJR allows a different response to negative shocks.",y=-3.45,size=21))
        self.finish("fhs_limits")

    def pit_theory(self):
        self.frame("Methods","PIT & Rosenblatt Transformation")
        theorem=panel("Probability Integral Transform",[
            eq(r"U=F_X(X)\sim\operatorname{Uniform}(0,1)",35),
            eq(r"X:\ \text{random return};\qquad F_X:\ \text{its true continuous CDF}",24),
            label("U is the return's percentile in its distribution: values between 0 and 1 are equally likely.",24),
        ],height=2.15).move_to([0,2.10,0])
        conditional=panel("Conditional PIT: the Rosenblatt transformation",[
            label("Evaluate each realised return using its conditional forecast CDF.",25),
            eq(r"u_t=\widehat F_{t\mid t-1}(r_t),\qquad t=1,\ldots,T",35),
            eq(r"r_t:\ \text{realised portfolio return on day }t",24),
            eq(r"\widehat F_{t\mid t-1}:\ \text{forecast CDF for day }t\text{, using information through day }t-1",24),
            eq(r"u_t:\ \text{percentile of the realised return within its forecast distribution}",24),
            eq(r"t:\ \text{evaluated day};\qquad T:\ \text{total number of evaluated forecasts}",24),
        ],height=3.55).move_to([0,-.95,0])
        dependence=left(label("Past returns inform each forecast, allowing volatility to change over time.",25),-6.94,-3.15)
        self.wait(.20)
        self.reveal(theorem)
        self.wait(.18)
        self.reveal(conditional)
        self.wait(.18)
        self.reveal(dependence)
        self.finish("pit_definition")
        self.frame("Methods","PIT · Calibration and Normal scores")
        uniform=panel("Under correct conditional forecasts",[
            label("If the continuous, one-step-ahead conditional forecast distributions are correct:",25),
            eq(r"\{u_t\}_{t=1}^{T}\overset{\mathrm{iid}}{\sim}\operatorname{Uniform}(0,1)",35),
            label("iid: independent and identically distributed.",24),
            label("Uniform percentiles, with no dependence over time.",24),
        ],height=2.35).move_to([0,1.99,0])
        normal=panel("Normal scores and Q–Q diagnostics",[
            eq(r"z_t=\Phi^{-1}(u_t)\overset{\mathrm{iid}}{\sim}N(0,1)\qquad\text{under the same assumption}",34),
            eq(r"z_t:\ \text{transformed score for day }t",24),
            eq(r"\Phi:\ \text{standard Normal CDF};\qquad\Phi^{-1}:\ \text{its inverse, or quantile function}",24),
            eq(r"N(0,1):\ \text{Normal distribution with mean }0\text{ and variance }1",24),
        ],height=2.2).move_to([0,-.70,0])
        diagnostic=bullet_list([
            "Compare ordered scores with standard Normal quantiles; departures reveal calibration errors.",
            "Uniformity concerns the distribution.",
            "Independence concerns patterns over time: test it separately."],13.88,24)
        left(diagnostic,-6.94,-2.30-diagnostic.height/2)
        normal.add(diagnostic)
        self.wait(.20)
        self.reveal(uniform)
        self.wait(.18)
        self.reveal(normal)
        self.finish("pit_theory")

    def data_slide(self):
        self.frame("Introduction","Data and portfolio")
        windows=block("Two evaluation windows · 11 euro-area equities",[
            "Stable: Jan 2014–Dec 2017 · 1,025 days     Stress: Jan 2018–Dec 2021 · 1,026 days",
            "Fixed portfolio weights · 250 past trading days per forecast · one-day horizon"],14.4,1.55,22).move_to([0,2.35,0])
        template=TexTemplate()
        template.add_to_preamble(r"\usepackage{lmodern,booktabs}")
        table=Tex((ROOT/'assets/portfolio_table.tex').read_text(encoding='utf-8'),
                  tex_template=template,tex_environment=None,color=INK).set_width(10.0).move_to([0,-1.03,0])
        if table.height>4.75:table.scale_to_fit_height(4.75)
        self.reveal(windows)
        self.play(DrawBorderThenFill(table),run_time=1.5)
        self.reveal(note("* Returns: percentage change in close prices. Source: Yahoo Finance. Weights sum to 100%.",y=-3.79,size=19))
        self.finish("data")

    def backtests(self,period):
        p=DATA[period];calm=period=="stable"
        years="2014–2017" if calm else "2018–2021"
        self.frame("Empirical results",f"Backtesting: {years}")
        self.reveal(note(f"{p['n']:,} daily forecasts · p-values · UC: frequency · IND: independence · CC: joint coverage",y=3.04,size=21))
        for c,y in ((95,1.58),(99,-.86)):
            card=Rectangle(width=14.4,height=2.20,stroke_width=0,fill_opacity=0).move_to([0,y,0])
            group=VGroup(card,left(label(f"{c}% VaR",26,NAVY,True),-6.92,y+.72))
            for x,text in zip((-3.8,.25,3.0,5.65),("Model","UC","IND","CC")):
                heading=label(text,22,NAVY,True)
                group.add(left(heading,x,y+.72) if text=='Model' else heading.move_to([x,y+.72,0]))
            group.add(Line([-3.8,y+.42,0],[6.8,y+.42,0],color=RULE,stroke_width=.85))
            group.add(Line([-6.92,y+1.08,0],[6.8,y+1.08,0],color=INK,stroke_width=1.0),
                      Line([-6.92,y-1.08,0],[6.8,y-1.08,0],color=INK,stroke_width=.8))
            for i,name in enumerate(("Normal","HS","FHS-GJR")):
                row_y=y+.18-i*.45
                group.add(left(label(name,24,bold=name=="FHS-GJR"),-3.8,row_y))
                for x,key in zip((.25,3.0,5.65),("p_uc","p_ind","p_cc")):
                    val=p["models"][name][str(c)][key]
                    obj=label("<.001" if val<.001 else f"{val:.3f}",24,bold=val<.05).move_to([x,row_y,0])
                    group.add(obj)
                    if val<.05:group.add(Line(obj.get_left()+DOWN*.2,obj.get_right()+DOWN*.2,color=INK,stroke_width=1.2))
            self.reveal(group)
        message="No model is rejected by these tests." if calm else "FHS-GJR alone avoids rejection in every test at both levels."
        self.reveal(left(label(message,26,NAVY),-7.2,-2.65),note("p: probability of equally or more extreme evidence under the null; underlined p < 0.05 ⇒ reject.",y=-3.25,size=20))
        if not calm:self.reveal(note("99% GJR Kupiec: p = 0.0533 · 17 breaches vs 10.26 expected — close to rejection.",y=-3.78,size=20))
        self.finish(period+"_tests")

    def return_chart(self,period):
        p=DATA[period]; calm=period=="stable"
        years="2014–2017" if calm else "2018–2021"
        self.frame("Empirical results",f"FHS-GJR · {years}")
        pick=(EMP["dates"]>=p["start"])&(EMP["dates"]<=p["end"])
        d=EMP["dates"][pick]; r=100*EMP["returns"][pick]; v=-100*EMP["gjr_var"][pick]
        lo=float(np.floor(min(r.min(),v.min())/5)*5); hi=float(np.ceil(r.max()/5)*5)
        hits=[r<v[:,k] for k in range(2)]
        counts=[int(h.sum()) for h in hits]
        legend=VGroup(legend_item("Daily returns",color=RETURNS,stroke_width=.9),
                      legend_item("95% VaR",color=ACCENT_COOL,stroke_width=2.6),
                      legend_item("99% VaR",color=ACCENT_WARM,stroke_width=3.0)).arrange(RIGHT,buff=.85).move_to([0,3.08,0])
        self.reveal(legend)
        ax=axes([0,len(d)-1,250],[lo,hi,5],12.5,4.95,-6.75,-2.72)
        ax.x_axis.set_opacity(0);ax.y_axis.set_opacity(0)
        group=VGroup(ax)
        for y in np.arange(lo,hi+.1,5):
            group.add(Line(ax.c2p(0,y),ax.c2p(len(d)-1,y),color="#E5E5E5",stroke_width=.55),
                      label(f"{y:g}%",18,MUTED).next_to(ax.c2p(0,y),LEFT,buff=.15))
        for year in range(2014 if calm else 2018,2018 if calm else 2022):
            index=int(np.searchsorted(d,f"{year}-01-01"))
            group.add(label(str(year),19,MUTED).next_to(ax.c2p(index,lo),DOWN,buff=.14))
        returns=line_on(ax,np.arange(len(d)),r,RETURNS,.9)
        var95=line_on(ax,np.arange(len(d)),v[:,0],ACCENT_COOL,2.6)
        var99=line_on(ax,np.arange(len(d)),v[:,1],ACCENT_WARM,3.0)
        for series in (returns,var95,var99):
            series.joint_type=LineJointType.ROUND
            series.set_cap_style(CapStyleType.ROUND)
        # A narrow white casing separates each VaR curve from the return trace.
        var95.set_stroke(color=BG,width=5.0,background=True)
        var99.set_stroke(color=BG,width=5.4,background=True)
        self.reveal(group)
        self.play(Create(returns),run_time=.85)
        self.play(Create(var95),Create(var99),run_time=1.25)
        end_points=[ax.c2p(len(d)-1,float(value)) for value in v[-1]]
        middle=(end_points[0][1]+end_points[1][1])/2
        separation=max(.48,end_points[0][1]-end_points[1][1])
        direct_labels=VGroup()
        for c,point,y,color in ((95,end_points[0],middle+separation/2,ACCENT_COOL),(99,end_points[1],middle-separation/2,ACCENT_WARM)):
            direct_labels.add(Line(point,[6.01,y,0],color=color,stroke_width=1.1),
                              left(label(f"{c}% VaR",20,color,bold=True),6.10,y))
        self.reveal(direct_labels)
        dots=VGroup(*[Dot(ax.c2p(int(i),float(r[i])),radius=.044,color=ACCENT_COOL).set_stroke(BG,width=1.0) for i in np.flatnonzero(hits[0])])
        rings=VGroup(*[VGroup(Circle(radius=.081,stroke_color=BG,stroke_width=3.6),
                            Circle(radius=.081,stroke_color=ACCENT_WARM,stroke_width=1.8)).move_to(ax.c2p(int(i),float(r[i]))) for i in np.flatnonzero(hits[1])])
        breach_legend=VGroup(legend_item(f"95% breaches: {counts[0]}",kind="dot",color=ACCENT_COOL,size=21),
                            legend_item(f"99% breaches: {counts[1]}",kind="circle",color=ACCENT_WARM,size=21)).arrange(RIGHT,buff=1.0).move_to([0,-3.48,0])
        self.play(FadeIn(dots),FadeIn(rings),FadeIn(breach_legend),run_time=.6)
        self.finish(period+"_returns",hold=1.7 if calm else .4)

    def qq_comparison(self):
        # The FHS-GJR points from the previous slide move into the right-hand panel.
        carry=self.qq_state['points']
        old_chrome=self.chrome
        self.remove(carry,old_chrome)
        outgoing=Group(*self.mobjects)
        self.clear();self.add(outgoing,carry)
        self.frame("Empirical results","Q–Q comparison · Crisis 2018–2021",clear=False)
        self.add(old_chrome)
        pick=crisis_mask(EMP)
        n=int(pick.sum()); theoretical=norm.ppf((np.arange(n)+.5)/n)
        group=VGroup()
        comparison_clouds=[]
        for x,name,prefix in ((-7.05,"Normal","normal"),(-2.05,"Historical Simulation","hs"),(2.95,"FHS-GJR","gjr")):
            ax=axes([-QQ_X_LIMIT,QQ_X_LIMIT,1],[-QQ_Y_LIMIT,QQ_Y_LIMIT,1],4.1,4.1,x,-2.0)
            ax.x_axis.set_opacity(0); ax.y_axis.set_opacity(0)
            group.add(fit(label(name,24,NAVY,True),4.35).move_to([x+2.05,2.49,0]))
            for value in (-12,-8,-4,0,4,8,12):
                group.add(Line(ax.c2p(-QQ_X_LIMIT,value),ax.c2p(QQ_X_LIMIT,value),color=RULE,stroke_width=.7),
                    label(str(value),16,MUTED).next_to(ax.c2p(-QQ_X_LIMIT,value),LEFT,buff=.09))
            for value in (-4,-2,0,2,4):
                group.add(label(str(value),16,MUTED).next_to(ax.c2p(value,-QQ_Y_LIMIT),DOWN,buff=.34))
            group.add(DashedLine(ax.c2p(-QQ_X_LIMIT,-QQ_X_LIMIT),ax.c2p(QQ_X_LIMIT,QQ_X_LIMIT),color=MUTED,stroke_width=1.2,dash_length=.09))
            raw_u,raw_scores=forecast_scores(EMP,prefix,pick)
            cloud=qq_cloud(ax,theoretical,np.sort(raw_scores),y_limit=QQ_Y_LIMIT)
            if prefix!='normal':
                lower=int((raw_u==0).sum());upper=int((raw_u==1).sum())
                group.add(qq_infinity_rows(ax,size=15))
                group.add(label(f"Outside support: {lower} below / {upper} above",17,MUTED).move_to([x+2.05,-2.82,0]))
            if prefix=='gjr':
                destination=cloud
                comparison_clouds.append(carry)
            else:
                group.add(cloud)
                comparison_clouds.append(cloud)
        group.add(label("Theoretical Normal quantiles",21).move_to([0,-3.42,0]),
                  label("Ordered Normal scores",18).rotate(PI/2).move_to([-7.75,.05,0]),
                  label(f"{n:,} observations",20).move_to([0,3.16,0]))
        self.play(FadeOut(outgoing),self.change_header(old_chrome),
                  Transform(carry,destination),FadeIn(group),run_time=1.45,rate_func=smooth)
        # Briefly highlight the tails (|theoretical quantile| >= 1.645) in all three panels.
        original_clouds=[cloud.copy() for cloud in comparison_clouds]
        muted_clouds=[cloud.copy() for cloud in comparison_clouds]
        for cloud in muted_clouds:
            for point in cloud:
                point.set_fill(opacity=point.get_fill_opacity()*.4)
                point.set_stroke(opacity=point.get_stroke_opacity()*.4)
        tail_focus=VGroup(*[point.copy() for cloud in comparison_clouds
            for point,value in zip(cloud,theoretical) if abs(value)>=1.645])
        for point in tail_focus:
            if point.get_fill_opacity()>0:point.set_fill(opacity=1)
        self.play(*[Transform(cloud,muted) for cloud,muted in zip(comparison_clouds,muted_clouds)],FadeIn(tail_focus),run_time=.6)
        self.wait(.5)
        self.play(*[Transform(cloud,original) for cloud,original in zip(comparison_clouds,original_clouds)],FadeOut(tail_focus),run_time=.6)
        self.finish("qq_comparison")

    def conclusions(self):
        self.frame("Conclusions","Findings")
        answer_text=label("FHS-GJR improves VaR forecasting under stress: alone not rejected in all tests at 95% and 99% (2018–2021).\nIn calm markets, tests do not discriminate (2014–2017); inference is specific to these data and windows.",25,INK)
        answer=VGroup(Rectangle(width=14.4,height=1.0,color=ACCENT_COOL,stroke_width=.9),answer_text).move_to([0,2.80,0])
        cards=VGroup(
            bullet_block("01 · Stable market",["No model is rejected: the tests do not discriminate among forecasts."],14.4,1.45,25),
            bullet_block("02 · Stress market",["Only asymmetric FHS-GJR avoids rejection at both 95% and 99%.","Leverage makes negative shocks raise forecast volatility more strongly."],14.4,1.85,25),
            bullet_block("03 · How convincing is the evidence?",["99% GJR Kupiec p = 0.0533: 17 breaches against 10.26 expected.","Borderline non-rejection supports caution, not definitive validation."],14.4,1.85,25)
        ).arrange(DOWN,buff=.22).move_to([0,-.77,0])
        self.reveal(answer)
        for card in cards:self.reveal(card)
        self.finish("conclusions")

    def limits(self):
        self.frame("Conclusions","Research limitations")
        def limitation_block(title,items,width,height,size):
            card=Rectangle(width=width,height=height,stroke_width=0,fill_opacity=0)
            heading=left(label(title,26,INK,True),-width/2+.24,height/2-.31)
            rule=Line([-width/2+.24,height/2-.64,0],[width/2-.24,height/2-.64,0],color=RULE,stroke_width=.8)
            points=(width-.48-(2*BULLET_RADIUS+BULLET_TEXT_GAP))/(.001037775*size)
            body=VGroup(*[Tex(r'\parbox[t]{'+f'{points:.4f}'+r'pt}{\raggedright '+tex_text(item)+r'\par}',
                             font_size=size,color=INK) for item in items])
            body.arrange(DOWN,buff=BULLET_ITEM_GAP,aligned_edge=LEFT)
            left(body,-width/2+.24,height/2-.84-body.height/2)
            return VGroup(card,rule,heading,body)
        cards=VGroup(
            limitation_block("01 · Information arrives after the shock",["Past returns only; no option-implied volatility to anticipate a regime change."],14.4,1.43,24),
            limitation_block("02 · Two calendar windows",["One stable and one stress period; the result need not generalise to other episodes."],14.4,1.43,24),
            limitation_block("03 · Concentrated portfolio and limited model set",["11 stocks, fixed weights, possible selection and survivorship bias.","Filtering is at portfolio level; time-varying asset correlations are not modelled."],14.4,1.88,24),
            limitation_block("04 · One-day validation",["Rare 99% breaches limit statistical power. Pooled Q–Q plots do not test independence.","Expected Shortfall (ES), the mean loss beyond VaR, is not backtested."],14.4,1.88,24)
        ).arrange(DOWN,buff=.18).move_to([0,-.37,0])
        for card in cards:self.reveal(card)
        self.finish("limits")

    def closing(self):
        self.clear(); self.number+=1
        def repository(url,x,y):
            icon=SVGMobject(ROOT/'assets/github.svg',height=.20).set_color(INK)
            return left(VGroup(icon,label(url.removeprefix('https://'),17,MUTED)).arrange(RIGHT,buff=.13),x,y)
        title=label('Thank you',68).move_to([0,2.85,0])
        top_rule=Line([-6.4,1.85,0],[6.4,1.85,0],color=RULE,stroke_width=.8)
        tool_heading=left(label('Animation & presentation',19,MUTED),-6.2,.72)
        manim=left(label('Manim',29),-6.2,.04)
        manim_link=repository(LINKS['manim'],-6.2,-.43)
        branch=VMobject(stroke_color='#999999',stroke_width=1.1).set_points_as_corners(
            [[-6.08,-.78,0],[-6.08,-1.17,0],[-5.60,-1.17,0]])
        slides=left(label('Manim Slides',27),-5.40,-1.17)
        slides_link=repository(LINKS['manim_slides'],-5.40,-1.63)
        slides_caption=left(label('Presentation extension for Manim',18,MUTED),-5.40,-2.04)
        project_heading=left(label('Code & research',19,MUTED),1.05,.72)
        project=left(label('Project repository',29),1.05,.04)
        project_link=repository(LINKS['project'],1.05,-.43)
        qr=SVGMobject(ROOT/'assets/repository_qr.svg',height=2.25).move_to([3.65,-1.70,0])
        project_caption=label('Repository: code, data and slides',20,MUTED).move_to([3.65,-2.91,0])
        # Centre the two credit columns at equal offsets from the middle.
        tool_column=VGroup(tool_heading,manim,manim_link,branch,slides,slides_link,slides_caption)
        tool_column.shift(RIGHT*(-3.65-tool_column.get_center()[0])+UP*.57)
        for item in (project_heading,project,project_link):
            item.set_x(3.65).shift(UP*.57)
        qr.shift(UP*.57)
        project_caption.shift(UP*.57)
        bottom_rule=Line([-6.4,-2.85,0],[6.4,-2.85,0],color=RULE,stroke_width=.8)
        codex_icon=SVGMobject(ROOT/'assets/chatgpt.svg',height=.36).set_color(INK)
        codex_credit=VGroup(codex_icon,label('Presentation coded using Codex',21,MUTED)).arrange(RIGHT,buff=.18).move_to([0,-3.35,0])
        self.play(FadeIn(title,shift=UP*.08),run_time=.7)
        self.play(GrowFromCenter(top_rule),run_time=.45)
        self.play(FadeIn(VGroup(tool_heading,manim,manim_link)),
                  FadeIn(VGroup(project_heading,project,project_link,qr,project_caption)),run_time=.65)
        self.play(Create(branch),FadeIn(VGroup(slides,slides_link,slides_caption)),run_time=.65)
        self.play(FadeIn(VGroup(bottom_rule,codex_credit)),run_time=.6)
        # Clickable areas and a vector copy of this page, used by build.py for the HTML and PDF.
        links=[dict(label=caption,url=LINKS[key],bounds=bounds(mob)) for key,caption,mob in
               [('manim','Manim on GitHub',manim_link),('manim_slides','Manim Slides on GitHub',slides_link),
                ('project','Project repository',project_link),('codex','OpenAI Codex',codex_credit)]]
        out=ROOT/'media';out.mkdir(exist_ok=True)
        (out/'closing_links.json').write_text(json.dumps(links,indent=2),encoding='utf-8')
        export_vector(self.mobjects,out/'closing_slide.svg')
        self.finish('closing',hold=1.4)


class CombinedQQ(Scene):
    """PDF-only page: the Normal Q-Q and tail-misspecification slides (12-13) in one static image."""
    def construct(self):
        self.camera.background_color=ManimColor(BG)
        self.add(Line([-7.4,3.63,0],[7.4,3.63,0],color=INK,stroke_width=.9),
                 left(fit(label('Normal Q–Q and tail misspecification',34,INK),14.8),-7.4,3.97),
                 label(f'12–13 / {TOTAL}',14,MUTED).move_to([7.0,-4.22,0]))
        scores=np.sort(simulated_pit()['scores']); n=len(scores)
        expected=norm.ppf((np.arange(n)+.5)/n)
        heavy=t.ppf((np.arange(n)+.5)/n,df=5)*t.pdf(0,df=5)/norm.pdf(0)
        ax=axes([-5,5,1],[-5,5,1],6.3,6.3,-3.15,-3.05); ax.set_opacity(0)
        grid=VGroup()
        for v in (-5,-3,0,3,5):
            grid.add(Line(ax.c2p(v,-5),ax.c2p(v,5),color=RULE,stroke_width=.7),Line(ax.c2p(-5,v),ax.c2p(5,v),color=RULE,stroke_width=.7),
                     label(str(v),17,MUTED).next_to(ax.c2p(v,-5),DOWN,buff=.1),label(str(v),17,MUTED).next_to(ax.c2p(-5,v),LEFT,buff=.1))
        def cloud(values,color):
            return VGroup(*[Dot(ax.c2p(float(x),float(y)),radius=.034 if abs(x)>1.645 else .021,color=color) for x,y in zip(expected,values)])
        legend=VGroup(legend_item('Approximate Normal\ncalibration','dot',ACCENT_COOL,size=23),
                      legend_item('Tail\nmisspecification','dot',ACCENT_WARM,size=23)).arrange(DOWN,buff=.38,aligned_edge=LEFT).move_to([-5.85,2.25,0])
        self.add(grid,Line(ax.c2p(-5,-5),ax.c2p(5,5),color=MUTED,stroke_width=1.4),cloud(heavy,ACCENT_WARM),cloud(scores,ACCENT_COOL),
                 label('Theoretical Normal quantiles',22).move_to([0,-3.65,0]),label('Ordered Normal scores',21).rotate(PI/2).move_to([-4.05,.1,0]),
                 legend,label('Heavier losses',26,ACCENT_WARM).move_to([-5.5,-1.4,0]),label('Heavier gains',26,ACCENT_WARM).move_to([5.3,2.0,0]))
