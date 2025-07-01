import re

def clean_html_table_from_file(input_file_path, output_file_path):
    # Read the input text file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Function to clean a <td> block
    def clean_td_block(td_text):
        # Keep only the first span with quick-nav-history-value
        match = re.search(r'<span class="quick-nav-history-value">([\d\.]+)</span>', td_text)
        if match:
            return match.group(1)
        else:
            # fallback: remove all tags and keep clean text
            clean_text = re.sub(r'<[^>]*>', '', td_text)
            return clean_text.strip()

    # Find all <tr> blocks
    rows = []
    trs = re.findall(r'<tr.*?>(.*?)</tr>', html_content, re.DOTALL)

    for tr in trs:
        tds = re.findall(r'<td.*?>(.*?)</td>', tr, re.DOTALL)
        cleaned_tds = [clean_td_block(td) for td in tds]
        if cleaned_tds:
            rows.append(cleaned_tds)

    # Build cleaned HTML
    final_html_rows = []
    for row in rows:
        tds_html = "".join(f"<td>{cell}</td>" for cell in row)
        final_html_rows.append(f"<tr>{tds_html}</tr>")

    final_cleaned_html = "\n".join(final_html_rows)

    # Save cleaned HTML to output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(final_cleaned_html)

    print(f"✅ Cleaned HTML table saved to {output_file_path}")

# Example usage:
clean_html_table_from_file('/Users/adipguduru/Downloads/table.txt', 'output.html')