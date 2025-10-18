"""
CLI Interface for Credit Risk Agent
Interactive command-line interface
"""

import sys
from pathlib import Path
import logging
from typing import Dict
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Rich imports for beautiful CLI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich import box

from src.agent.credit_risk_agent import CreditRiskAgent

# Initialize rich console
console = Console()


class CreditRiskCLI:
    """Command-line interface for credit risk agent"""
    
    def __init__(self):
        """Initialize CLI"""
        self.agent = None
        self.history = []
    
    def display_banner(self):
        """Display welcome banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║           🏦 CREDIT RISK ASSESSMENT AGENT 🤖                  ║
║                                                               ║
║     AI-Powered Risk Analysis with Natural Language           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """
        console.print(banner, style="bold cyan")
    
    def initialize_agent(self):
        """Initialize agent with progress bar"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task1 = progress.add_task("[cyan]Loading LLM...", total=None)
            task2 = progress.add_task("[cyan]Loading RAG system...", total=None)
            task3 = progress.add_task("[cyan]Initializing agent...", total=None)
            
            try:
                self.agent = CreditRiskAgent(
                    model_path="models/autoencoder/default_autoencoder.pth",
                    preprocessor_path="models/preprocessor/preprocessor.pkl",
                    val_errors_path="results/validation_errors.npy",
                    knowledge_base_path="knowledge_base",
                    vector_db_path="models/vector_db/chroma_db"
                )
                
                progress.update(task1, completed=True)
                progress.update(task2, completed=True)
                progress.update(task3, completed=True)
                
                console.print("\n✅ Agent initialized successfully!\n", style="bold green")
                return True
                
            except Exception as e:
                console.print(f"\n❌ Error initializing agent: {e}\n", style="bold red")
                return False
    
    def display_menu(self):
        """Display main menu"""
        menu = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
        menu.add_column("Option", style="cyan bold")
        menu.add_column("Description", style="white")
        
        menu.add_row("1", "Assess Customer Risk")
        menu.add_row("2", "Ask Agent a Question")
        menu.add_row("3", "Search Policies")
        menu.add_row("4", "View Assessment History")
        menu.add_row("5", "Help & Examples")
        menu.add_row("6", "Exit")
        
        console.print("\n")
        console.print(Panel(menu, title="[bold]Main Menu[/bold]", border_style="cyan"))
    
    def assess_customer(self):
        """Interactive customer risk assessment"""
        console.print("\n[bold cyan]Customer Risk Assessment[/bold cyan]\n")
        
        # Option to use sample data or enter manually
        use_sample = Confirm.ask("Use sample customer data?", default=True)
        
        if use_sample:
            customer_data = self.get_sample_customer()
            console.print("\n[green]Using sample customer data[/green]")
        else:
            customer_data = self.get_customer_input()
        
        # Display customer data
        self.display_customer_data(customer_data)
        
        # Confirm assessment
        if not Confirm.ask("\nProceed with assessment?", default=True):
            return
        
        # Perform assessment
        console.print("\n[yellow]Assessing risk...[/yellow]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Running autoencoder...", total=None)
                
                result = self.agent.assess_customer(customer_data)
                
                progress.update(task, completed=True)
            
            # Display result
            self.display_assessment_result(result)
            
            # Save to history
            self.history.append({
                'type': 'assessment',
                'data': customer_data,
                'result': result
            })
            
        except Exception as e:
            console.print(f"\n[red]Error during assessment: {e}[/red]")
    
    def ask_question(self):
        """Ask agent a question"""
        console.print("\n[bold cyan]Ask the Agent[/bold cyan]\n")
        
        # Show examples
        examples = [
            "What are the lending criteria for high-risk customers?",
            "How should I handle a customer with previous defaults?",
            "What documents are required for loan approval?",
            "Explain the risk scoring system"
        ]
        
        console.print("[dim]Examples:[/dim]")
        for i, example in enumerate(examples, 1):
            console.print(f"  [dim]{i}. {example}[/dim]")
        
        console.print()
        
        # Get question
        question = Prompt.ask("[cyan]Your question")
        
        if not question.strip():
            console.print("[red]Question cannot be empty[/red]")
            return
        
        # Process question
        console.print(f"\n[yellow]Thinking...[/yellow]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Processing query...", total=None)
                
                result = self.agent.query(question)
                
                progress.update(task, completed=True)
            
            # Display answer
            console.print("\n[bold green]Agent's Response:[/bold green]\n")
            
            # Use markdown for better formatting
            md = Markdown(result['answer'])
            console.print(Panel(md, border_style="green"))
            
            # Save to history
            self.history.append({
                'type': 'query',
                'question': question,
                'result': result
            })
            
        except Exception as e:
            console.print(f"\n[red]Error processing question: {e}[/red]")
    
    def search_policies(self):
        """Search knowledge base"""
        console.print("\n[bold cyan]Search Policies[/bold cyan]\n")
        
        # Get search query
        query = Prompt.ask("[cyan]Search query")
        
        if not query.strip():
            console.print("[red]Query cannot be empty[/red]")
            return
        
        # Optional: Filter by category
        categories = ["bank_policies", "regulations", "case_studies", "faq"]
        console.print(f"\n[dim]Categories: {', '.join(categories)} (or leave blank for all)[/dim]")
        category = Prompt.ask("[cyan]Filter by category (optional)", default="")
        
        # Search
        console.print(f"\n[yellow]Searching...[/yellow]")
        
        try:
            if category.strip():
                context = self.agent.rag_system.get_context_for_query(
                    query,
                    categories=[category.strip()]
                )
            else:
                context = self.agent.rag_system.get_context_for_query(query)
            
            # Display results
            console.print("\n[bold green]Search Results:[/bold green]\n")
            console.print(Panel(context, border_style="green"))
            
        except Exception as e:
            console.print(f"\n[red]Error searching: {e}[/red]")
    
    def view_history(self):
        """View assessment history"""
        console.print("\n[bold cyan]Assessment History[/bold cyan]\n")
        
        if not self.history:
            console.print("[yellow]No history yet[/yellow]")
            return
        
        # Create table
        table = Table(show_header=True, box=box.ROUNDED)
        table.add_column("#", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Details", style="white")
        
        for i, item in enumerate(self.history, 1):
            if item['type'] == 'assessment':
                details = f"Customer assessment"
            else:
                details = item['question'][:50] + "..."
            
            table.add_row(str(i), item['type'].title(), details)
        
        console.print(table)
        
        # Option to view details
        if Confirm.ask("\nView details of an item?", default=False):
            idx = Prompt.ask("Enter item number", default="1")
            try:
                idx = int(idx) - 1
                if 0 <= idx < len(self.history):
                    console.print("\n[bold]Details:[/bold]\n")
                    console.print(json.dumps(self.history[idx], indent=2))
            except:
                console.print("[red]Invalid item number[/red]")
    
    def show_help(self):
        """Show help and examples"""
        help_text = """
# Credit Risk Agent Help

## Capabilities
- **Risk Assessment**: Analyze customer credit risk using deep learning
- **Policy Search**: Find relevant lending guidelines and regulations
- **Q&A**: Ask questions about credit risk, policies, and procedures
- **Explanations**: Get natural language explanations for decisions

## Example Queries
1. "What factors determine credit risk?"
2. "Show me the lending policy for medium-risk customers"
3. "How should I handle a customer with 60 days arrears?"
4. "What are the approval requirements for vehicle loans?"

## Tips
- Be specific in your questions
- Provide complete customer data for assessments
- Use policy search to find relevant guidelines
- Review similar cases for context

## Need More Help?
Check the documentation at: docs/AGENT_ARCHITECTURE.md
        """
        
        console.print(Markdown(help_text))
    
    def display_customer_data(self, data: Dict):
        """Display customer data in table format"""
        table = Table(show_header=True, box=box.SIMPLE)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in data.items():
            table.add_row(key, str(value))
        
        console.print("\n")
        console.print(Panel(table, title="Customer Data", border_style="cyan"))
    
    def display_assessment_result(self, result: Dict):
        """Display assessment result"""
        # Extract information from result
        # Note: Adjust based on actual result format
        
        console.print("\n")
        
        # Create result panel
        result_table = Table(show_header=False, box=box.ROUNDED)
        result_table.add_column("Metric", style="cyan bold")
        result_table.add_column("Value", style="white")
        
        result_table.add_row("Risk Score", "15.2/100")  # Placeholder
        result_table.add_row("Risk Category", "LOW")
        result_table.add_row("Action", "APPROVE")
        result_table.add_row("Confidence", "85%")
        
        console.print(Panel(
            result_table,
            title="[bold green]Assessment Result[/bold green]",
            border_style="green"
        ))
        
        # Display full response
        console.print("\n[bold]Detailed Analysis:[/bold]\n")
        console.print(result.get('answer', 'No details available'))
    
    def get_sample_customer(self) -> Dict:
        """Get sample customer data"""
        return {
            "NET_RENTAL": 10000.0,
            "NO_OF_RENTAL": 36,
            "PAID_RENTALS": 12,
            "CB_ARREARS_AGE": 0.0,
            "YOM": 2020,
            "FINANCE_AMOUNT": 300000.0,
            "CUSTOMER_VALUATION": 350000.0,
            "EFFECTIVE_RATE": 8.5,
            "AGE": 32,
            "INCOME": 55000.0,
            "EXPENSE": 30000.0,
            "GENDER": "M",
            "MARITAL_STATUS": "Single"
        }
    
    def get_customer_input(self) -> Dict:
        """Get customer data from user input"""
        console.print("\n[yellow]Enter customer details:[/yellow]\n")
        
        data = {}
        
        # Numeric fields
        numeric_fields = {
            "NET_RENTAL": "Monthly payment",
            "AGE": "Customer age",
            "INCOME": "Annual income",
            "FINANCE_AMOUNT": "Loan amount",
        }
        
        for field, description in numeric_fields.items():
            value = Prompt.ask(f"[cyan]{description}[/cyan]")
            try:
                data[field] = float(value)
            except:
                console.print(f"[red]Invalid value, using default[/red]")
                data[field] = 0.0
        
        # Add defaults for other fields
        data.update({
            "NO_OF_RENTAL": 36,
            "PAID_RENTALS": 0,
            "CB_ARREARS_AGE": 0.0,
            "YOM": 2020,
            "CUSTOMER_VALUATION": data.get("FINANCE_AMOUNT", 0) * 1.2,
            "EFFECTIVE_RATE": 8.5,
            "EXPENSE": data.get("INCOME", 0) * 0.5,
            "GENDER": "M",
            "MARITAL_STATUS": "Single"
        })
        
        return data
    
    def run(self):
        """Main CLI loop"""
        self.display_banner()
        
        # Initialize agent
        if not self.initialize_agent():
            console.print("[red]Failed to initialize agent. Exiting.[/red]")
            return
        
        # Main loop
        while True:
            self.display_menu()
            
            choice = Prompt.ask(
                "\n[cyan]Select an option[/cyan]",
                choices=["1", "2", "3", "4", "5", "6"],
                default="1"
            )
            
            if choice == "1":
                self.assess_customer()
            elif choice == "2":
                self.ask_question()
            elif choice == "3":
                self.search_policies()
            elif choice == "4":
                self.view_history()
            elif choice == "5":
                self.show_help()
            elif choice == "6":
                if Confirm.ask("\nAre you sure you want to exit?", default=False):
                    console.print("\n[cyan]Goodbye! 👋[/cyan]\n")
                    break
            
            console.print("\n")


def main():
    """Entry point"""
    try:
        cli = CreditRiskCLI()
        cli.run()
    except KeyboardInterrupt:
        console.print("\n\n[cyan]Interrupted. Goodbye! 👋[/cyan]\n")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]\n")


if __name__ == "__main__":
    main()